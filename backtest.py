"""Walk-forward / out-of-sample backtest of the Extreme-Mover regime classifier (v2).

Timeline for every signal
  t      : features computed from data through the close of t
  t+1    : entry at the CLOSE (close-only execution)
  t+2... : exits evaluated on closes only (stop / time exit)

Model training: monthly walk-forward, expanding window. For each test month M the models
are fit only on candidate events whose full label horizon ended >= EMBARGO sessions before
the first session of M (purged + embargoed). Regime thresholds come from TRAINING base rates.

Research protocol: OOS months Jul-2023..Jun-2025 = DEVELOPMENT (design choices were made
looking at these). Jul-2025..Oct-2026 = HOLDOUT, reported separately and not tuned on.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from features import ROOT, load_panels, load_meta, FEATURE_COLS
from icw_model import ICComposite
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Pre-specified, economically motivated inputs for the probability scores
LR_CONT = ["ret_60", "dist_sma200", "accel_20v60", "z_3", "clv_mean5", "n_close_high5", "rvol_mean10", "vol_accel",
           "idio_z_20", "xs_peer_20", "event_hold", "off_high10", "rsi14", "ext_atr_sma20", "mkt_rv20"]
LR_REV = ["ret_5", "ret_60", "z_5", "rvol_max5", "vol_fade", "failed_high", "wick_mean3", "intraday_rev",
          "confirm_break", "break_prev_low", "ext_atr_sma20", "dist_vwap20", "steepening", "macd_hist_fade",
          "idio_share_20", "rv20", "mkt_rv20"]


class LRProb:
    def __init__(self, cols, label):
        self.cols, self.label = cols, label

    def fit(self, tr):
        tr = tr.dropna(subset=[self.label])
        self.med = tr[self.cols].median()
        X = tr[self.cols].fillna(self.med).clip(tr[self.cols].quantile(0.01), tr[self.cols].quantile(0.99), axis=1)
        self.lo, self.hi = tr[self.cols].quantile(0.01), tr[self.cols].quantile(0.99)
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500)).fit(X, tr[self.label])
        self.base = tr[self.label].mean()
        return self

    def predict(self, df):
        X = df[self.cols].fillna(self.med).clip(self.lo, self.hi, axis=1)
        return self.m.predict_proba(X)[:, 1]

    def coefs(self):
        return pd.Series(self.m[-1].coef_[0], index=self.cols).round(3)


def blowoff(df):
    return ((df.ret_5 >= 0.40) & (df.rvol_max5 >= 5) & ((df.failed_high > 0) | (df.wick_mean3 > 0.35))
            & (df.confirm_break > 0))


def fresh_breakout(df):
    return (df.ret_60 <= 0.50) & (df.accel_20v60 > 0) & (df.clv_mean5 >= 0.5) & ~blowoff(df)


CFG = dict(
    min_price=3.0, min_dollar_vol=15e6,
    hold_long=10, hold_short=5,
    stop_long_atr=2.0, short_stop_spike_buf_atr=0.25, short_stop_min_atr=1.0, short_stop_max_atr=2.5,
    slip_bps=10, slip_illiquid_bps=20, illiquid_dv=50e6,
    win_thr=0.03,
    label_horizon=11, embargo=5,
    first_test="2023-07-01", holdout_start="2025-07-01",
    lift=1.25, margin=0.05, min_exh=3, min_cont=3,
    max_pos=10, max_side=7, max_new_per_day=3,
    risk_per_trade=0.0125, cap_long=0.15, cap_short=0.10,
    icw_window_months=12,
)


def candidate_mask(df):
    move = ((df.ret_1 >= 0.12) | (df.ret_3 >= 0.20) | (df.ret_5 >= 0.20) | (df.ret_10 >= 0.30)
            | (df.ret_20 >= 0.30) | (df.ret_60 >= 0.50) | (df.EMS >= 40))
    liq = (df.price >= CFG["min_price"]) & (df.dollar_vol20 >= CFG["min_dollar_vol"])
    return move & liq


def borrow_rate(rv, price):
    return np.where(rv > 1.5, 0.60, np.where((rv > 1.0) | (price < 10), 0.25, 0.05))


def slippage(dv):
    return np.where(dv < CFG["illiquid_dv"], CFG["slip_bps"] + CFG["slip_illiquid_bps"], CFG["slip_bps"]) / 1e4


# ------------------------------------------------------------- forward outcomes
def forward_outcomes(cand, C, cfg=CFG):
    dates = C.index
    di = {d: i for i, d in enumerate(dates)}
    sym_i = {s: j for j, s in enumerate(C.columns)}
    A = C.values
    n = len(dates)
    keys = ["entry", "entry_i", "long_ret", "long_days", "long_exit_i", "short_ret", "short_days",
            "short_exit_i", "stop_long", "stop_short", "fwd10", "min5", "complete"]
    out = {k: np.full(len(cand), np.nan) for k in keys}
    ti, sj = cand.date.map(di).values, cand.symbol.map(sym_i).values
    atr, spike = cand.ATR.values, cand.spike_high5.values
    for r in range(len(cand)):
        i, j = ti[r], sj[r]
        if i + 1 >= n or not np.isfinite(A[i + 1, j]):
            continue
        e = A[i + 1, j]
        out["entry"][r], out["entry_i"][r] = e, i + 1
        if i + 11 < n:
            out["fwd10"][r] = A[i + 11, j] / e - 1
            out["min5"][r] = np.nanmin(A[i + 2:i + 7, j]) / e - 1
            out["complete"][r] = 1
        # long: close-based stop at entry - 2 ATR, time exit after hold_long sessions
        stop = e - cfg["stop_long_atr"] * atr[r]
        out["stop_long"][r] = stop
        last, kx = e, None
        for k in range(1, cfg["hold_long"] + 1):
            if i + 1 + k >= n:
                break
            c = A[i + 1 + k, j]
            if not np.isfinite(c):
                continue
            last, kx = c, k
            if c < stop:
                break
        if kx is not None:
            out["long_ret"][r], out["long_days"][r], out["long_exit_i"][r] = last / e - 1, kx, i + 1 + kx
        # short: invalidation = close back above the spike high (+buffer), bounded 1.0-2.5 ATR
        stop = min(max(spike[r] + cfg["short_stop_spike_buf_atr"] * atr[r], e + cfg["short_stop_min_atr"] * atr[r]),
                   e + cfg["short_stop_max_atr"] * atr[r])
        out["stop_short"][r] = stop
        last, kx = e, None
        for k in range(1, cfg["hold_short"] + 1):
            if i + 1 + k >= n:
                break
            c = A[i + 1 + k, j]
            if not np.isfinite(c):
                continue
            last, kx = c, k
            if c > stop:
                break
        if kx is not None:
            out["short_ret"][r], out["short_days"][r], out["short_exit_i"][r] = 1 - last / e, kx, i + 1 + kx
    o = pd.DataFrame(out, index=cand.index)
    sl = slippage(cand.dollar_vol20.values)
    o["slip"] = sl
    o["long_net"] = o.long_ret - 2 * sl
    o["borrow"] = borrow_rate(cand.rv20.fillna(1).values, cand.price.values)
    o["short_net"] = o.short_ret - 2 * sl - o.borrow * o.short_days.fillna(0) / 252
    o["stop_dist_long"] = (o.entry - o.stop_long) / o.entry
    o["stop_dist_short"] = (o.stop_short - o.entry) / o.entry
    full_l = o.long_exit_i.notna() & ((o.long_days == cfg["hold_long"]) | (o.long_ret < -o.stop_dist_long + 1e-12))
    full_s = o.short_exit_i.notna() & ((o.short_days == cfg["hold_short"]) | (o.short_ret < -o.stop_dist_short + 1e-12))
    o["y_cont"] = np.where(full_l, (o.long_net > cfg["win_thr"]).astype(float), np.nan)
    o["y_rev"] = np.where(full_s, (o.short_net > cfg["win_thr"]).astype(float), np.nan)
    return o


# -------------------------------------------------------------- walk-forward fit
def make_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=250,
                                          min_samples_leaf=60, l2_regularization=1.0, random_state=7)


def walk_forward(ds, dates):
    di = {d: i for i, d in enumerate(dates)}
    ds = ds.assign(di=ds.date.map(di))
    preds, folds = [], []
    for m in pd.period_range(CFG["first_test"], ds.date.max(), freq="M"):
        test = ds[(ds.date >= m.start_time) & (ds.date <= m.end_time)]
        if test.empty:
            continue
        cutoff = test.di.min() - CFG["label_horizon"] - CFG["embargo"]
        tr_c = ds[(ds.di <= cutoff) & ds.y_cont.notna()]
        tr_r = ds[(ds.di <= cutoff) & ds.y_rev.notna()]
        mc, mr = make_model(), make_model()
        mc.fit(tr_c[FEATURE_COLS], tr_c.y_cont)
        mr.fit(tr_r[FEATURE_COLS], tr_r.y_rev)
        p = test[["date", "symbol"]].copy()
        p["p_cont"] = mc.predict_proba(test[FEATURE_COLS])[:, 1]
        p["p_rev"] = mr.predict_proba(test[FEATURE_COLS])[:, 1]
        p["base_cont"], p["base_rev"] = tr_c.y_cont.mean(), tr_r.y_rev.mean()
        if CFG["icw_window_months"]:
            lo = (m.start_time - pd.DateOffset(months=CFG["icw_window_months"]))
            wc, wr = tr_c[tr_c.date >= lo], tr_r[tr_r.date >= lo]
        else:
            wc, wr = tr_c, tr_r
        ic_c = ICComposite("long_net", "y_cont").fit(wc)
        ic_r = ICComposite("short_net", "y_rev").fit(wr)
        p["s_cont"], p["q_cont"], p["ev_long"] = ic_c.predict(test)
        p["s_rev"], p["q_rev"], p["ev_short"] = ic_r.predict(test)
        p["qbase_cont"], p["qbase_rev"] = ic_c.base, ic_r.base
        lo18 = m.start_time - pd.DateOffset(months=18)
        lr_c = LRProb(LR_CONT, "y_cont").fit(tr_c[tr_c.date >= lo18])
        lr_r = LRProb(LR_REV, "y_rev").fit(tr_r[tr_r.date >= lo18])
        p["mom_prob"], p["rev_prob"] = lr_c.predict(test), lr_r.predict(test)
        p["mom_base"], p["rev_base"] = lr_c.base, lr_r.base
        preds.append(p)
        last_icw = (ic_c, ic_r, lr_c, lr_r)
        folds.append(dict(month=str(m), n_train=len(tr_c), n_test=len(test), train_end=str(dates[cutoff].date()),
                          base_cont=round(tr_c.y_cont.mean(), 3), base_rev=round(tr_r.y_rev.mean(), 3)))
    return pd.concat(preds), pd.DataFrame(folds), (mc, mr, last_icw)


def classify(df, mode="ml"):
    if mode == "ml":
        mom = ((df.p_cont >= CFG["lift"] * df.base_cont) & (df.p_cont >= df.p_rev + CFG["margin"])
               & (df.cont_count >= CFG["min_cont"]))
        rev = ((df.p_rev >= CFG["lift"] * df.base_rev) & (df.p_rev >= df.p_cont + CFG["margin"])
               & (df.exh_count >= CFG["min_exh"]) & (df.confirm_break > 0))
        em, er = df.p_cont / df.base_cont, df.p_rev / df.base_rev
    elif mode == "final":
        mom = fresh_breakout(df) & (df.mom_prob >= df.mom_base)
        rev = blowoff(df)
        em, er = df.mom_prob, df.rev_prob + 1.0   # shorts are rare: give them priority for a slot
    elif mode == "icw":
        mom = ((df.ev_long >= 0.015) & (df.q_cont >= CFG["lift"] * df.qbase_cont) & (df.ev_long > df.ev_short)
               & (df.cont_count >= 2))
        rev = ((df.ev_short >= 0.010) & (df.q_rev >= CFG["lift"] * df.qbase_rev) & (df.ev_short > df.ev_long)
               & (df.exh_count >= 2))
        em, er = df.ev_long, df.ev_short
    else:  # rules only, thresholds fixed a priori (no fitting)
        mom = (df.MOM_rule >= 65) & (df.REV_rule < 45) & (df.cont_count >= 4)
        rev = (df.REV_rule >= 55) & (df.exh_count >= 4) & (df.MOM_rule < 55) & (df.confirm_break > 0)
        em, er = df.MOM_rule / 65, df.REV_rule / 55
    regime = np.where(mom & ~rev, "MOMENTUM", np.where(rev & ~mom, "REVERSAL", "NO TRADE"))
    edge = np.where(regime == "MOMENTUM", em, np.where(regime == "REVERSAL", er, np.nan))
    return pd.Series(regime, index=df.index), pd.Series(edge, index=df.index)


def build_signals(df, regime, edge):
    s = df.assign(regime=regime, edge=edge)
    s = s[s.regime != "NO TRADE"].dropna(subset=["entry_i"]).copy()
    isl = s.regime == "MOMENTUM"
    s["exit_i"] = np.where(isl, s.long_exit_i, s.short_exit_i)
    s["net_ret"] = np.where(isl, s.long_net, s.short_net)
    s["stop_dist"] = np.where(isl, s.stop_dist_long, s.stop_dist_short)
    s["borrow"] = np.where(isl, 0.0, s.borrow)
    return s.dropna(subset=["exit_i"])


# ------------------------------------------------------------------ portfolio sim
def simulate(sig, C, dates, start=None, end=None, cfg=CFG):
    A = C.values
    sym_i = {s: j for j, s in enumerate(C.columns)}
    if start is not None:
        sig = sig[(sig.date >= start) & (sig.date <= end)]
    sig = sig.sort_values(["date", "edge"], ascending=[True, False])
    by_entry = {}
    for r in sig.itertuples():
        by_entry.setdefault(int(r.entry_i), []).append(r)
    if sig.empty:
        e = pd.DataFrame({"equity": [1.0, 1.0], "n_pos": [0, 0], "gross": [0.0, 0.0]},
                         index=pd.to_datetime([start or dates[0], end or dates[-1]]))
        return e, pd.DataFrame(columns=["symbol", "regime", "signal_date", "entry_date", "exit_date", "weight", "net_ret"])
    i0 = int(sig.entry_i.min())
    i1 = len(dates) - 1 if end is None else min(len(dates) - 1, int(sig.exit_i.max()))
    equity, curve, open_pos, trades = 1.0, [], [], []
    for i in range(i0, i1 + 1):
        pnl, still = 0.0, []
        for p in open_pos:
            c0, c1 = A[i - 1, p["j"]], A[i, p["j"]]
            if np.isfinite(c0) and np.isfinite(c1):
                d = (c1 / c0 - 1) * p["mv"]
                pnl += d * p["side"]
                p["mv"] *= c1 / c0
            if p["side"] == -1:
                pnl -= p["borrow"] / 252 * p["mv"]
            if i >= p["exit_i"]:
                pnl -= p["slip"] * p["mv"]
                trades.append(dict(symbol=p["sym"], regime=p["regime"], signal_date=p["sd"], entry_date=dates[p["ei"]],
                                   exit_date=dates[i], weight=p["w"], net_ret=p["net"]))
            else:
                still.append(p)
        open_pos = still
        equity += pnl
        new, held = 0, {p["sym"] for p in open_pos}
        for r in by_entry.get(i, []):
            if len(open_pos) >= cfg["max_pos"] or new >= cfg["max_new_per_day"]:
                break
            side = 1 if r.regime == "MOMENTUM" else -1
            if r.symbol in held or sum(p["side"] == side for p in open_pos) >= cfg["max_side"]:
                continue
            cap = cfg["cap_long"] if side == 1 else cfg["cap_short"]
            w = min(cap, cfg["risk_per_trade"] / max(r.stop_dist, 1e-3))
            mv = w * equity
            equity -= r.slip * mv
            open_pos.append(dict(sym=r.symbol, j=sym_i[r.symbol], side=side, regime=r.regime, sd=r.date, ei=i,
                                 exit_i=int(r.exit_i), mv=mv, w=w, slip=r.slip, borrow=r.borrow, net=r.net_ret))
            held.add(r.symbol)
            new += 1
        gross = sum(p["mv"] for p in open_pos) / equity if equity > 0 else np.nan
        curve.append((dates[i], equity, len(open_pos), gross))
    eq = pd.DataFrame(curve, columns=["date", "equity", "n_pos", "gross"]).set_index("date")
    return eq, pd.DataFrame(trades)


# --------------------------------------------------------------------- metrics
def trade_stats(r):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return dict(n=0)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=int(len(r)), win_rate=float((r > 0).mean()), avg=float(r.mean()), median=float(r.median()),
                profit_factor=float(wins / losses) if losses > 0 else float("inf"),
                best=float(r.max()), worst=float(r.min()))


def equity_stats(eq):
    r = eq.equity.pct_change().dropna()
    dd = eq.equity / eq.equity.cummax() - 1
    yrs = max(len(r) / 252, 1e-9)
    roll63 = eq.equity.pct_change(63).dropna()
    return dict(start=str(eq.index[0].date()), end=str(eq.index[-1].date()),
                total_return=float(eq.equity.iloc[-1] / eq.equity.iloc[0] - 1),
                cagr=float((eq.equity.iloc[-1] / eq.equity.iloc[0]) ** (1 / yrs) - 1),
                sharpe=float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
                max_drawdown=float(dd.min()), ann_vol=float(r.std() * np.sqrt(252)),
                avg_positions=float(eq.n_pos.mean()) if "n_pos" in eq else np.nan,
                avg_gross=float(eq.gross.mean()) if "gross" in eq else np.nan,
                pct_3m_windows_positive=float((roll63 > 0).mean()) if len(roll63) else np.nan,
                median_3m_return=float(roll63.median()) if len(roll63) else np.nan,
                worst_3m_return=float(roll63.min()) if len(roll63) else np.nan,
                best_3m_return=float(roll63.max()) if len(roll63) else np.nan)


def period_block(oos, sigs, C, dates, start, end):
    o = oos[(oos.date >= start) & (oos.date <= end)]
    res = {}
    for mode, sig in sigs.items():
        s = sig[(sig.date >= start) & (sig.date <= end)]
        eq, tr = simulate(sig, C, dates, start, end)
        res[mode] = dict(
            signals=dict(MOMENTUM=trade_stats(s[s.regime == "MOMENTUM"].net_ret),
                         REVERSAL=trade_stats(s[s.regime == "REVERSAL"].net_ret)),
            portfolio_trades=dict(MOMENTUM=trade_stats(tr[tr.regime == "MOMENTUM"].net_ret),
                                  REVERSAL=trade_stats(tr[tr.regime == "REVERSAL"].net_ret),
                                  ALL=trade_stats(tr.net_ret)),
            equity=equity_stats(eq))
        # long-only / short-only portfolio sims to report max DD + Sharpe per side
        for side in ["MOMENTUM", "REVERSAL"]:
            ss = sig[sig.regime == side]
            if len(ss[(ss.date >= start) & (ss.date <= end)]) > 5:
                e2, _ = simulate(ss, C, dates, start, end)
                res[mode][f"equity_{side}_only"] = equity_stats(e2)
        eq.to_csv(f"{ROOT}/out/equity_{mode}_{start[:7]}.csv")
        tr.to_csv(f"{ROOT}/out/trades_{mode}_{start[:7]}.csv", index=False)
    res["baselines"] = dict(long_all_candidates=trade_stats(o.long_net), short_all_candidates=trade_stats(o.short_net),
                            short_all_confirmed_breaks=trade_stats(o[o.confirm_break > 0].short_net),
                            short_RSI80=trade_stats(o[o.rsi14 >= 80].short_net),
                            long_top_decile_ret20=trade_stats(o[o.rs_rank20 >= 0.9].long_net),
                            long_fresh_breakout_gate_only=trade_stats(o[fresh_breakout(o)].long_net),
                            short_blowoff_gate=trade_stats(o[blowoff(o)].short_net))
    spy = C["SPY"].loc[start:end]
    res["SPY"] = equity_stats(pd.DataFrame({"equity": spy / spy.iloc[0]}))
    x = o.dropna(subset=["y_cont"])
    y = o.dropna(subset=["y_rev"])
    res["auc"] = dict(mom_prob=float(roc_auc_score(x.y_cont, x.mom_prob)), rev_prob=float(roc_auc_score(y.y_rev, y.rev_prob)),
                      cont_icw=float(roc_auc_score(x.y_cont, x.s_cont)), rev_icw=float(roc_auc_score(y.y_rev, y.s_rev)),
                      cont_ml=float(roc_auc_score(x.y_cont, x.p_cont)), cont_rule=float(roc_auc_score(x.y_cont, x.MOM_rule.fillna(0))),
                      rev_ml=float(roc_auc_score(y.y_rev, y.p_rev)), rev_rule=float(roc_auc_score(y.y_rev, y.REV_rule.fillna(0))),
                      rev_rsi=float(roc_auc_score(y.y_rev, y.rsi14.fillna(50))))
    return res


def main():
    os.makedirs(f"{ROOT}/out", exist_ok=True)
    P = load_panels()
    C, dates = P["close"], P["close"].index
    feats = pd.read_pickle(f"{ROOT}/data/features.pkl")
    feats = feats[feats.symbol.isin(load_meta().query("group != 'etf'").index)]
    cand = feats[candidate_mask(feats)].reset_index(drop=True)
    out = forward_outcomes(cand, C)
    ds = cand.join(out)
    ds = ds[ds.date >= "2022-09-01"]
    preds, folds, models = walk_forward(ds, dates)
    oos = ds.merge(preds, on=["date", "symbol"]).copy()
    sigs = {}
    for mode in ["final", "icw", "ml", "rules"]:
        reg, edge = classify(oos, mode)
        oos[f"regime_{mode}"] = reg
        oos[f"edge_{mode}"] = edge
        sigs[mode] = build_signals(oos, reg, edge)
        sigs[mode].to_csv(f"{ROOT}/out/signals_{mode}.csv", index=False)
    end = str(oos.date.max().date())
    results = dict(
        n_candidates_total=int(len(cand)), n_oos=int(len(oos)),
        development=period_block(oos, sigs, C, dates, CFG["first_test"], "2025-06-30"),
        holdout=period_block(oos, sigs, C, dates, CFG["holdout_start"], end),
        full=period_block(oos, sigs, C, dates, CFG["first_test"], end),
        folds=folds.to_dict("records"), cfg=CFG)
    cal = {}
    for col, y in [("p_cont", "y_cont"), ("p_rev", "y_rev")]:
        x = oos.dropna(subset=[y])
        q = pd.qcut(x[col], 5, labels=False, duplicates="drop")
        cal[col] = x.groupby(q).agg(pred=(col, "mean"), actual=(y, "mean"), n=(y, "size")).round(3).to_dict("records")
    results["calibration"] = cal
    ic_c, ic_r, lr_c, lr_r = models[2]
    results["lr_final_coefs"] = dict(momentum=lr_c.coefs().to_dict(), reversal=lr_r.coefs().to_dict())
    results["icw_final_weights"] = dict(cont=ic_c.w.sort_values().round(4).to_dict(), rev=ic_r.w.sort_values().round(4).to_dict(),
                                        cont_t=ic_c.ic_t.round(2).to_dict(), rev_t=ic_r.ic_t.round(2).to_dict())
    import pickle
    pickle.dump(models, open(f"{ROOT}/out/models_last_fold.pkl", "wb"))
    oos.to_pickle(f"{ROOT}/out/oos.pkl")
    json.dump(results, open(f"{ROOT}/out/backtest_results.json", "w"), indent=1, default=float)
    return results


def summarize(r, periods=("development", "holdout", "full")):
    def f(x):
        return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"
    for per in periods:
        b = r[per]
        print(f"\n######## {per.upper()}  SPY: ret {f(b['SPY']['total_return'])} sharpe {f(b['SPY']['sharpe'])} mdd {f(b['SPY']['max_drawdown'])}")
        print("AUC", {k: round(v, 3) for k, v in b["auc"].items()})
        for mode in ["final", "icw", "ml", "rules"]:
            m = b[mode]
            e = m["equity"]
            print(f"  [{mode}] equity ret {f(e['total_return'])} cagr {f(e['cagr'])} sharpe {f(e['sharpe'])} mdd {f(e['max_drawdown'])} "
                  f"3m+ {f(e['pct_3m_windows_positive'])} med3m {f(e['median_3m_return'])} gross {f(e['avg_gross'])}")
            for side in ["MOMENTUM", "REVERSAL"]:
                s = m["signals"][side]
                if s.get("n"):
                    print(f"     {side:9s} signals n={s['n']:5d} win {f(s['win_rate'])} avg {f(s['avg'])} med {f(s['median'])} PF {f(s['profit_factor'])}", end="")
                    eo = m.get(f"equity_{side}_only")
                    print(f" | solo sharpe {f(eo['sharpe'])} mdd {f(eo['max_drawdown'])}" if eo else "")
        for k, s in b["baselines"].items():
            print(f"  base {k:28s} n={s['n']:6d} win {f(s['win_rate'])} avg {f(s['avg'])} med {f(s['median'])} PF {f(s['profit_factor'])}")


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        if a.startswith("--win="):
            CFG["icw_window_months"] = int(a.split("=")[1]) or None
    if "--dev-only" in sys.argv:
        summarize(main(), ("development",))
    else:
        summarize(main())
