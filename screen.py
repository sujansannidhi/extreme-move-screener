"""Daily screen: score every liquid stock as of the latest close and emit the ranked table.

Run after the close (features need the full day's bar). Entries are for the NEXT session's close.
Uses the models from the most recent walk-forward fold (trained only on events whose outcomes
were fully known before that fold).
"""
import pickle
import numpy as np
import pandas as pd

from features import ROOT, load_meta
from backtest import candidate_mask, blowoff, fresh_breakout, CFG, LRProb  # noqa: F401 (needed to unpickle)
from icw_model import ICComposite  # noqa: F401

meta = load_meta()
feats = pd.read_pickle(f"{ROOT}/data/features.pkl")
asof = feats.date.max()
today = feats[(feats.date == asof) & feats.symbol.isin(meta.query("group != 'etf'").index)].copy()
mc, mr, (ic_c, ic_r, lr_c, lr_r) = pickle.load(open(f"{ROOT}/out/models_last_fold.pkl", "rb"))

liq = (today.price >= CFG["min_price"]) & (today.dollar_vol20 >= CFG["min_dollar_vol"])
today = today[liq].copy()
today["candidate"] = candidate_mask(today)
today["peer_share_20"] = 1 - today.idio_share_20
today["MomentumScore"] = 100 * lr_c.predict(today)
today["ReversalScore"] = 100 * lr_r.predict(today)
today["mom_base"], today["rev_base"] = 100 * lr_c.base, 100 * lr_r.base
is_mom = fresh_breakout(today) & (today.MomentumScore >= today.mom_base) & today.candidate
is_rev = blowoff(today) & today.candidate
today["regime"] = np.where(is_rev, "REVERSAL", np.where(is_mom, "MOMENTUM", "NO TRADE"))
today["direction"] = today.regime.map({"MOMENTUM": "LONG", "REVERSAL": "SHORT", "NO TRADE": "—"})


def why_no_trade(r):
    if not r.candidate:
        return "move not extreme enough"
    reasons = []
    if r.ret_60 > 0.50:
        reasons.append(f"already extended (+{r.ret_60:.0%} in 60d)")
    if r.accel_20v60 <= 0:
        reasons.append("momentum decelerating")
    if r.clv_mean5 < 0.5:
        reasons.append("weak closes")
    if r.MomentumScore < r.mom_base and not reasons:
        reasons.append("momentum score below base rate")
    if r.ReversalScore >= 40:
        reasons.append("high pullback risk")
    if r.ret_5 >= 0.40 and not r.confirm_break:
        reasons.append("blow-off not yet breaking")
    return "; ".join(reasons) or "ambiguous"


def setup_text(r):
    if r.regime == "MOMENTUM":
        s = "Fresh breakout"
        if r.event_recent and r.event_hold:
            s += ", gap held"
        if r.idio_share_20 >= 0.7:
            s += ", stock-specific"
        elif r.peer_share_20 >= 0.5:
            s += ", group rally"
        return s
    if r.regime == "REVERSAL":
        return "Parabolic blow-off, breaking"
    return "No trade: " + why_no_trade(r)


def levels(r):
    atr = r.ATR
    if r.regime == "MOMENTUM":
        stop = r.price - CFG["stop_long_atr"] * atr
        return (f"Buy at Mon 10/5 close (MOC)",
                f"Close < ${stop:,.2f} (entry − 2 ATR); re-anchor to actual fill", "10 sessions")
    if r.regime == "REVERSAL":
        stop = min(max(r.spike_high5 + 0.25 * atr, r.price + atr), r.price + 2.5 * atr)
        return (f"Short at Mon 10/5 close (MOC) if still below ${r.high_t:,.2f}",
                f"Close > ${stop:,.2f} (above spike high)", "5 sessions")
    return ("—", "—", "—")


today["setup"] = today.apply(setup_text, axis=1)
lv = today.apply(levels, axis=1, result_type="expand")
today[["entry_condition", "invalidation", "holding_period"]] = lv
today["move_type"] = np.where(today.peer_share_20 >= 0.5, "industry-wide",
                              np.where(today.idio_share_20 >= 0.7, "company-specific", "mixed"))
today = today.join(meta[["sector", "subindustry", "peer_group", "market_cap", "pe_ratio", "float"]], on="symbol")
today["turnover_pct"] = 100 * today.turnover
cols = ["symbol", "direction", "regime", "setup", "EMS", "MomentumScore", "ReversalScore", "ret_1", "ret_5", "ret_20",
        "ret_60", "rvol", "rvol_max5", "rv20", "atr_pct", "turnover_pct", "xs_peer_20", "xs_sector_20", "peer_ret_20",
        "idio_z_20", "move_type", "peer_group", "dist_sma20", "dist_sma50", "dist_sma200", "dist_52wh", "rsi14",
        "gap", "exh_count", "cont_count", "price", "ATR", "high_t", "low_t", "spike_high5", "market_cap", "pe_ratio",
        "entry_condition", "invalidation", "holding_period", "candidate"]
out = today[cols].sort_values(["candidate", "EMS"], ascending=False)
out.to_csv(f"{ROOT}/out/screen_{asof.date()}.csv", index=False)
print("as of", asof.date(), "| liquid:", len(today), "| candidates:", int(today.candidate.sum()),
      "| MOMENTUM:", int(is_mom.sum()), "| REVERSAL:", int(is_rev.sum()))
show = ["symbol", "regime", "EMS", "MomentumScore", "ReversalScore", "ret_1", "ret_5", "ret_20", "ret_60", "rvol", "move_type", "setup"]
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
print(out[out.regime != "NO TRADE"].sort_values("MomentumScore", ascending=False)[show].round(3).to_string(index=False))
print(out[(out.regime == "NO TRADE") & out.candidate].head(25)[show].round(3).to_string(index=False))
