"""Feature engine for the Extreme-Mover screener.

Every feature for date t uses only OHLCV data up to and including the close of t.
Trades are entered no earlier than the close of t+1, so all features are lagged by at
least one full session relative to the entry price. Baseline volatility / beta / volume
references are additionally shifted back (pre-move) so a move cannot inflate its own
normalizer.
"""
import numpy as np
import pandas as pd

import os
ROOT = os.path.dirname(os.path.abspath(__file__))
HORIZONS = [1, 3, 5, 10, 20, 60]
# corporate-action artifacts in the vendor history: drop data before these dates
ARTIFACT_CUTS = {"HUT": "2023-12-05", "CORZ": "2024-01-25"}


# ----------------------------------------------------------------------------- data
def load_panels(path=f"{ROOT}/data/ohlcv_full.csv.gz"):
    d = pd.read_csv(path, parse_dates=["date"])
    for s, dt in ARTIFACT_CUTS.items():
        d = d[~((d.symbol == s) & (d.date < dt))]
    P = {c: d.pivot(index="date", columns="symbol", values=c).sort_index()
         for c in ["open", "high", "low", "close", "volume"]}
    return P


def load_meta():
    u = pd.read_csv(f"{ROOT}/data/universe.csv").set_index("symbol")
    f = pd.read_csv(f"{ROOT}/data/fundamentals.csv").set_index("symbol")
    m = u.join(f[["shares_outstanding", "float", "market_cap", "pe_ratio", "industry"]], how="left")
    # peer group: GICS/curated sub-industry when it has >=4 members, else sector
    stocks = m[m.group != "etf"]
    cnt = stocks.subindustry.value_counts()
    m["peer_group"] = np.where(m.subindustry.map(cnt).fillna(0) >= 4, m.subindustry, m.sector)
    return m


# ------------------------------------------------------------------------ helpers
def wilder(x, n):
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(C, n=14):
    d = C.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def clip01(x):
    return x.clip(lower=0, upper=1) if isinstance(x, (pd.DataFrame, pd.Series)) else np.clip(x, 0, 1)


def peer_excess(R, groups):
    """Mean horizon return of the *other* members of each stock's peer group."""
    out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
    for g, cols in groups.items():
        cols = [c for c in cols if c in R.columns]
        if len(cols) < 2:
            continue
        sub = R[cols]
        s, n = sub.sum(axis=1, min_count=1), sub.notna().sum(axis=1)
        other_sum = (-sub).add(s, axis=0)
        other_n = (-sub.notna().astype(int)).add(n, axis=0)
        out[cols] = other_sum / other_n.where(other_n > 0)
    return out


# ----------------------------------------------------------------------- features
def compute_features(P, meta):
    O, H, L, C, V = P["open"], P["high"], P["low"], P["close"], P["volume"]
    F = {}
    lr = np.log(C / C.shift(1))
    Cp = C.shift(1)

    # --- returns & acceleration
    for n in HORIZONS:
        F[f"ret_{n}"] = C / C.shift(n) - 1
    slope = {n: np.log(C / C.shift(n)) / n for n in [3, 5, 10, 20, 60]}
    F["accel_3v10"] = slope[3] - slope[10]
    F["accel_5v20"] = slope[5] - slope[20]
    F["accel_20v60"] = slope[20] - slope[60]
    # increasingly steep returns: count of ordered slopes s3>s5>s10>s20
    F["steepening"] = ((slope[3] > slope[5]).astype(int) + (slope[5] > slope[10]).astype(int)
                       + (slope[10] > slope[20]).astype(int)).where(C.notna())

    # --- moving averages / extension
    sma = {n: C.rolling(n, min_periods=int(n * 0.75)).mean() for n in [20, 50, 200]}
    for n in [20, 50, 200]:
        F[f"dist_sma{n}"] = C / sma[n] - 1
    F["sma20_slope"] = sma[20] / sma[20].shift(5) - 1
    F["sma50_slope"] = sma[50] / sma[50].shift(10) - 1
    F["ma_align"] = ((sma[20] > sma[50]).astype(int) + (sma[50] > sma[200]).astype(int)
                     + (F["sma20_slope"] > 0).astype(int) + (F["sma50_slope"] > 0).astype(int)).where(sma[50].notna())
    hi52 = H.rolling(252, min_periods=120).max()
    F["dist_52wh"] = C / hi52 - 1

    # --- volatility
    TR = pd.concat([H - L, (H - Cp).abs(), (L - Cp).abs()]).groupby(level=0).max().reindex(C.index)
    TR = TR[C.columns]
    ATR = wilder(TR, 14)
    F["atr_pct"] = ATR / C
    F["ext_atr_sma20"] = (C - sma[20]) / ATR
    F["atr_expansion"] = TR.rolling(5).mean() / TR.rolling(50, min_periods=30).mean()
    F["rv20"] = lr.rolling(20).std() * np.sqrt(252)
    F["rv_ratio"] = lr.rolling(10).std() / lr.rolling(60, min_periods=40).std()
    base_vol = lr.rolling(60, min_periods=40).std().shift(20)       # pre-move daily vol
    F["base_vol"] = base_vol
    sd20 = C.rolling(20).std()
    F["bb_pctb"] = (C - (sma[20] - 2 * sd20)) / (4 * sd20)
    F["bb_width"] = 4 * sd20 / sma[20]

    # --- oscillators
    F["rsi14"] = rsi(C, 14)
    ema12, ema26 = C.ewm(span=12, adjust=False).mean(), C.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    F["macd_n"] = macd / C
    F["macd_hist_n"] = hist / C
    F["macd_hist_fade"] = (hist / hist.rolling(5).max()).where(hist.rolling(5).max() > 0)  # <1 = fading from peak

    # --- candle structure
    rng = (H - L).replace(0, np.nan)
    F["gap"] = O / Cp - 1
    F["range_pct"] = (H - L) / Cp
    F["clv"] = ((C - L) / rng).fillna(0.5)
    F["upper_wick"] = ((H - np.maximum(O, C)) / rng).fillna(0)
    F["clv_mean5"] = F["clv"].rolling(5).mean()
    F["n_close_high5"] = (F["clv"] > 0.7).rolling(5).sum()
    F["wick_mean3"] = F["upper_wick"].rolling(3).mean()
    F["gap_max5"] = F["gap"].rolling(5).max()
    TP = (H + L + C) / 3
    vwap20 = (TP * V).rolling(20).sum() / V.rolling(20).sum()
    F["dist_vwap20"] = C / vwap20 - 1

    # --- volume
    vavg20 = V.rolling(20, min_periods=15).mean().shift(1)
    rvol = V / vavg20
    F["rvol"] = rvol
    F["rvol_max5"] = rvol.rolling(5).max()
    F["rvol_mean10"] = rvol.rolling(10).mean()
    F["vol_accel"] = V.rolling(5).mean() / vavg20
    lv = np.log(V.replace(0, np.nan))
    F["abn_vol_z"] = (lv - lv.rolling(60, min_periods=40).mean().shift(1)) / lv.rolling(60, min_periods=40).std().shift(1)
    rmax10 = rvol.rolling(10).max()
    days_since_vpeak = rvol.rolling(10).apply(lambda x: len(x) - 1 - np.nanargmax(x) if np.isfinite(x).any() else np.nan, raw=True)
    F["days_since_vol_peak"] = days_since_vpeak
    F["vol_fade"] = (rvol.rolling(3).mean() / rmax10).where(days_since_vpeak >= 2)  # low = volume dried up after spike
    dv = C * V
    F["dollar_vol20"] = dv.rolling(20, min_periods=15).mean()
    so = meta.shares_outstanding.reindex(C.columns)
    F["turnover"] = V.div(so, axis=1)                 # uses current share count (slow-moving)
    F["turnover20"] = F["turnover"].rolling(20).mean()

    # --- extreme-move z-scores vs pre-move volatility
    for n in HORIZONS:
        F[f"z_{n}"] = np.log(C / C.shift(n)) / (base_vol * np.sqrt(n))

    # --- industry / sector adjustment
    stocks = meta[meta.group != "etf"].index.intersection(C.columns)
    groups = {g: list(ix) for g, ix in meta.loc[stocks].groupby("peer_group").groups.items()}
    etf_map = meta.sector_etf.reindex(C.columns)
    for n in [1, 5, 20, 60]:
        R = F[f"ret_{n}"]
        peer = peer_excess(R[stocks], groups).reindex(columns=C.columns)
        etf_ret = pd.DataFrame({s: R[e] if e in R.columns else np.nan for s, e in etf_map.items()}, index=C.index)
        F[f"peer_ret_{n}"] = peer
        F[f"xs_peer_{n}"] = R - peer
        F[f"xs_sector_{n}"] = R - etf_ret
    # beta-adjusted residual (beta estimated on 120d ending 20d ago)
    etf_lr = pd.DataFrame({s: lr[e] if e in lr.columns else np.nan for s, e in etf_map.items()}, index=C.index)
    cov = (lr * etf_lr).rolling(120, min_periods=80).mean() - lr.rolling(120, min_periods=80).mean() * etf_lr.rolling(120, min_periods=80).mean()
    beta = (cov / etf_lr.rolling(120, min_periods=80).var()).shift(20).clip(-1, 4)
    resid = lr - beta * etf_lr
    rvb = resid.rolling(60, min_periods=40).std().shift(20)
    for n in [1, 5, 20]:
        F[f"idio_z_{n}"] = resid.rolling(n).sum() / (rvb * np.sqrt(n))
    # share of the 20d move explained by the peer group
    F["peer_share_20"] = clip01(F["peer_ret_20"] / F["ret_20"].where(F["ret_20"] > 0.02))
    F["idio_share_20"] = 1 - F["peer_share_20"]
    F["peer_share_5"] = clip01(F["peer_ret_5"] / F["ret_5"].where(F["ret_5"] > 0.01))
    # group breadth: fraction of peers up >15% over 20d (industry-wide speculative rally)
    up = (F["ret_20"] > 0.15).astype(float).where(F["ret_20"].notna())
    breadth = peer_excess(up[stocks], groups).reindex(columns=C.columns)
    F["peer_breadth_20"] = breadth
    F["rs_rank20"] = F["ret_20"][stocks].rank(axis=1, pct=True).reindex(columns=C.columns)
    F["rs_rank60"] = F["ret_60"][stocks].rank(axis=1, pct=True).reindex(columns=C.columns)

    # --- event structure (catalyst proxy from price/volume)
    lr20 = np.log(C / C.shift(20))
    big_day = lr.rolling(20).max()
    F["event_share"] = clip01(big_day / lr20.where(lr20 > 0.02))     # 1 = whole move came in one session
    ev_flag = (F["gap"] > 0.05) & (rvol > 3)
    ev_low = L.where(ev_flag).ffill(limit=10)
    ev_close = C.where(ev_flag).ffill(limit=10)
    F["event_recent"] = ev_flag.rolling(10).max()
    F["event_hold"] = ((C > ev_close) & (L.rolling(3).min() > ev_low)).astype(float).where(ev_low.notna(), np.nan)

    # --- exhaustion evidence
    hh10 = H.shift(1).rolling(10).max()
    new_high = H >= hh10
    F["failed_high"] = (new_high & (F["clv"] < 0.35)).astype(float).rolling(3).max()
    F["intraday_rev"] = (((F["gap"] > 0.02) & (C < O)) | (new_high & (F["clv"] < 0.3))).astype(float).rolling(3).max()
    F["off_high10"] = C / C.rolling(10).max() - 1
    rsi_ = F["rsi14"]
    F["rsi_div"] = ((C >= 0.98 * C.rolling(10).max()) & (rsi_ < rsi_.rolling(10).max() - 8)).astype(float)
    F["pullback_atr10"] = (C.rolling(10).max() - C) / ATR
    # breakdown confirmation (the move has actually started to fail)
    ema8 = C.ewm(span=8, adjust=False).mean()
    F["dist_ema8"] = C / ema8 - 1
    F["break_prev_low"] = (C < L.shift(1)).astype(float).where(C.notna())
    F["red_day"] = (C < O).astype(float).where(C.notna())
    F["lower_close2"] = ((C < C.shift(1)) & (C.shift(1) < C.shift(2))).astype(float).where(C.notna())
    F["days_since_high10"] = C.rolling(10).apply(lambda x: len(x) - 1 - np.nanargmax(x) if np.isfinite(x).any() else np.nan, raw=True)
    F["confirm_break"] = (((F["break_prev_low"] > 0) & (F["red_day"] > 0))
                          | ((F["lower_close2"] > 0) & (F["days_since_high10"] <= 4))).astype(float).where(C.notna())
    # market regime (same value for every stock on a date)
    if "SPY" in C.columns:
        spy = C["SPY"]
        mk = {"mkt_ret20": spy / spy.shift(20) - 1,
              "mkt_dist50": spy / spy.rolling(50).mean() - 1,
              "mkt_rv20": np.log(spy / spy.shift(1)).rolling(20).std() * np.sqrt(252)}
        if "IWM" in C.columns:
            mk["smallcap_ret20"] = C["IWM"] / C["IWM"].shift(20) - 1
        for k, v in mk.items():
            F[k] = pd.DataFrame(np.repeat(v.values[:, None], C.shape[1], axis=1), index=C.index, columns=C.columns)

    # --- continuation structure
    bo = H.rolling(60, min_periods=40).max().shift(21)          # resistance before the move
    F["above_breakout"] = C / bo - 1
    F["breakout_hold"] = C.rolling(10).min() / bo - 1
    F["price"] = C
    F["ATR"] = ATR
    F["spike_high5"] = H.rolling(5).max()
    F["low_t"] = L
    F["high_t"] = H

    return F, ATR


# -------------------------------------------------------------------- rule scores
def rule_scores(F):
    """Interpretable 0-100 composites (no fitting)."""
    ret_thr = {1: 0.20, 3: 0.30, 5: 0.40, 10: 0.50, 20: 0.75, 60: 1.50}
    A = pd.concat([clip01(F[f"z_{n}"] / 6.0) for n in HORIZONS]).groupby(level=0).max()
    B = pd.concat([clip01(F[f"ret_{n}"] / t) for n, t in ret_thr.items()]).groupby(level=0).max()
    A, B = A.reindex(F["price"].index), B.reindex(F["price"].index)
    Cv = clip01(np.log(F["rvol_max5"].clip(lower=1)) / np.log(10))
    D = clip01(F["atr_expansion"] - 1)
    E = clip01(F["idio_z_20"].clip(lower=0) / 5)
    # volume/vol/idio components only count when there is an actual upside move
    up_gate = clip01(2 * np.maximum(A, B))
    EMS = 100 * (0.30 * A + 0.30 * B + (0.15 * Cv + 0.10 * D + 0.15 * E) * up_gate)

    exh = (0.12 * clip01(F["ext_atr_sma20"] / 6) + 0.08 * clip01(F["dist_vwap20"] / 0.4)
           + 0.10 * (F["steepening"] / 3) * (F["accel_5v20"] > 0)
           + 0.10 * clip01(np.log(F["rvol_max5"].clip(lower=1)) / np.log(8))
           + 0.10 * clip01(1 - F["vol_fade"].fillna(1) / 0.5)
           + 0.05 * clip01(F["gap_max5"] / 0.25)
           + 0.06 * F["failed_high"] + 0.05 * F["intraday_rev"] + 0.04 * clip01(F["wick_mean3"] / 0.5)
           + 0.06 * clip01(1 - F["macd_hist_fade"].fillna(1)) + 0.04 * F["rsi_div"]
           + 0.05 * clip01((F["rsi14"] - 70) / 20)
           + 0.08 * F["idio_share_20"].fillna(0.5)
           + 0.07 * clip01((F["rv20"] - 0.6) / 1.0))
    REV = 100 * exh

    mom = (0.15 * F["rs_rank20"].fillna(0.5)
           + 0.10 * clip01((F["rvol_mean10"] - 1) / 2)
           + 0.10 * (F["n_close_high5"] / 5)
           + 0.15 * clip01(F["breakout_hold"] / 0.05 + 0.5)
           + 0.15 * (F["ma_align"] / 4)
           + 0.10 * clip01(1 + F["off_high10"] / 0.15)
           + 0.10 * F["event_hold"].fillna(0.5)
           + 0.15 * (1 - exh.clip(0, 1)))
    MOM = 100 * mom
    return EMS, MOM, REV


def exhaustion_evidence(F):
    """Count of discrete parabolic/exhaustion flags (0-8)."""
    flags = [F["ext_atr_sma20"] > 4, F["dist_vwap20"] > 0.25, F["steepening"] >= 2,
             F["rvol_max5"] > 4, F["vol_fade"] < 0.4, F["failed_high"] > 0,
             F["intraday_rev"] > 0, F["wick_mean3"] > 0.35, F["macd_hist_fade"] < 0.7, F["rsi_div"] > 0]
    out = sum(f.astype(int) for f in flags)
    return out.where(F["price"].notna())


def continuation_evidence(F):
    flags = [F["rs_rank20"] > 0.9, F["rvol_mean10"] > 1.5, F["n_close_high5"] >= 3,
             F["breakout_hold"] > 0, F["ma_align"] >= 3, F["off_high10"] > -0.08,
             F["event_hold"] > 0]
    out = sum(f.astype(int) for f in flags)
    return out.where(F["price"].notna())


FEATURE_COLS = (
    [f"ret_{n}" for n in HORIZONS] + [f"z_{n}" for n in HORIZONS]
    + ["accel_3v10", "accel_5v20", "accel_20v60", "steepening", "dist_sma20", "dist_sma50", "dist_sma200",
       "sma20_slope", "sma50_slope", "ma_align", "dist_52wh", "atr_pct", "ext_atr_sma20", "atr_expansion",
       "rv20", "rv_ratio", "bb_pctb", "bb_width", "rsi14", "macd_n", "macd_hist_n", "macd_hist_fade",
       "gap", "range_pct", "clv", "upper_wick", "clv_mean5", "n_close_high5", "wick_mean3", "gap_max5",
       "dist_vwap20", "rvol", "rvol_max5", "rvol_mean10", "vol_accel", "abn_vol_z", "days_since_vol_peak",
       "vol_fade", "turnover20", "xs_peer_1", "xs_peer_5", "xs_peer_20", "xs_peer_60", "xs_sector_5",
       "xs_sector_20", "xs_sector_60", "idio_z_1", "idio_z_5", "idio_z_20", "idio_share_20", "peer_share_5",
       "peer_breadth_20", "rs_rank20", "rs_rank60", "event_share", "event_recent", "event_hold",
       "failed_high", "intraday_rev", "off_high10", "rsi_div", "pullback_atr10", "above_breakout",
       "breakout_hold", "dist_ema8", "break_prev_low", "red_day", "lower_close2", "days_since_high10",
       "confirm_break", "mkt_ret20", "mkt_dist50", "mkt_rv20", "smallcap_ret20",
       "EMS", "MOM_rule", "REV_rule", "exh_count", "cont_count"]
)


def build_long_table(P, meta):
    F, ATR = compute_features(P, meta)
    EMS, MOM, REV = rule_scores(F)
    F["EMS"], F["MOM_rule"], F["REV_rule"] = EMS, MOM, REV
    F["exh_count"], F["cont_count"] = exhaustion_evidence(F), continuation_evidence(F)
    keep = FEATURE_COLS + ["price", "ATR", "dollar_vol20", "spike_high5", "low_t", "high_t", "turnover",
                           "peer_ret_20", "peer_ret_5", "base_vol"]
    long = pd.concat({k: F[k].stack(future_stack=True) for k in keep}, axis=1)
    long.index.names = ["date", "symbol"]
    long = long[long["price"].notna()]
    return long.reset_index()


if __name__ == "__main__":
    import time
    t = time.time()
    P = load_panels()
    meta = load_meta()
    df = build_long_table(P, meta)
    df.to_parquet(f"{ROOT}/data/features.parquet") if False else df.to_pickle(f"{ROOT}/data/features.pkl")
    print(df.shape, f"{time.time()-t:.1f}s")
    print(df[df.date == df.date.max()].sort_values("EMS", ascending=False)
          [["symbol", "EMS", "MOM_rule", "REV_rule", "ret_1", "ret_5", "ret_20", "ret_60", "rvol", "exh_count"]].head(25).to_string())
