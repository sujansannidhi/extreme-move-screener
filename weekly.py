"""Weekly model check (runs Saturdays via .github/workflows/weekly.yml).

  1. Accuracy check: grade the current long rule's last 40 closed trades against the backtest expectation
     (about 83% profitable, average +0.3% to +0.6%). Status: ON TRACK / WATCH / ALERT. ALERT pauses longs;
     the next ON TRACK week resumes them.
  2. Re-tune (learning from recent results): pick the target multiple and hold length that, over the last
     26 weeks, kept at least 80% of trades profitable and had the best average. This guarded re-tuning matched
     the fixed rule in a 2023–2026 walk-forward test (out/adaptive_compare.csv); unguarded re-tuning was unstable.
  3. Short re-test: shorts switch on only if the short rule over the last 52 weeks has avg ≥ +0.5%,
     ≥ 80% profitable and no single loss worse than −35%.
  4. Mistake review: what the last 13 weeks' losing trades had in common, versus winners (descriptive only).
Writes out/weekly/<date>.json, updates model/params.json and appends model/changelog.md when something changes.
"""
import itertools
import json
import os

import numpy as np
import pandas as pd

from features import ROOT, load_panels, load_meta
from backtest import candidate_mask
from nextday import build_paths
from strategy import load_params, simulate, exit_cost, borrow_rate, PARAMS_PATH

GRID = list(itertools.product([0.25, 0.5, 0.75], [3, 5]))
EXPECT = dict(win=0.83, avg_lo=0.003, avg_hi=0.006)
DESCRIBE = {"ret_60": "60-day gain", "ret_20": "20-day gain", "ret_5": "5-day gain", "atr_pct": "daily range (ATR%)",
            "REV_rule": "reversal score", "rvol": "relative volume", "dist_sma20": "distance above 20-day average",
            "mkt_ret20": "S&P 500 20-day return"}


def rule_outcomes(side, rank_by, combos):
    P = load_panels()
    meta = load_meta()
    f = pd.read_pickle(f"{ROOT}/data/features.pkl")
    f = f[f.symbol.isin(meta.query("group != 'etf'").index) & (f.symbol != "SOXL")]
    f = f[(f.price >= 3) & (f.dollar_vol20 >= 15e6) & f.atr_pct.notna()]
    f = f[candidate_mask(f)]
    pk = f.sort_values([rank_by, "dollar_vol20"], ascending=False).groupby("date").head(3).reset_index(drop=True)
    E, paths, ed = build_paths(pk, P, 1)
    cost = exit_cost(pk.dollar_vol20.values)
    br = borrow_rate(pk.rv20.fillna(1).values, pk.price.values)
    dates = P["close"].index
    di = np.searchsorted(dates.values, ed)
    for m, H in combos:
        r, d, _, _ = simulate(E, pk.atr_pct.values, paths["O"], paths["H"], paths["L"], paths["C"], side, m, H, cost, br)
        pk[f"r_{m}_{H}"] = r
        known = np.where(np.isfinite(d), di + np.nan_to_num(d).astype(int), len(dates) + 10)
        pk[f"k_{m}_{H}"] = [dates[k] if k < len(dates) else pd.NaT for k in known]
    return pk, dates


def tstats(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), win=round(float((x > 0).mean()), 3), avg=round(float(x.mean()), 5),
                worst=round(float(x.min()), 4), p05=round(float(x.quantile(0.05)), 4))


def main():
    params = load_params()
    old = json.loads(json.dumps(params))
    today = pd.Timestamp.today().normalize()
    report = dict(date=str(today.date()), params_before=old, actions=[])
    lp = params["long"]

    # ---- long rule outcomes for the whole grid
    pk, dates = rule_outcomes("long", lp["rank_by"], GRID)
    asof = dates[-1]
    cur = f"{lp['target_atr']}_{lp['max_days']}" if (lp["target_atr"], lp["max_days"]) in GRID else None
    if cur is None:
        pk2, _ = rule_outcomes("long", lp["rank_by"], [(lp["target_atr"], lp["max_days"])])
        pk[f"r_{lp['target_atr']}_{lp['max_days']}"] = pk2[f"r_{lp['target_atr']}_{lp['max_days']}"]
        pk[f"k_{lp['target_atr']}_{lp['max_days']}"] = pk2[f"k_{lp['target_atr']}_{lp['max_days']}"]
        cur = f"{lp['target_atr']}_{lp['max_days']}"

    # 1) accuracy check on the last 40 closed trades of the current rule
    closed = pk[pk[f"k_{cur}"].notna()].sort_values(f"k_{cur}")
    last40 = closed.tail(40)[f"r_{cur}"]
    s40 = tstats(last40)
    if s40.get("n", 0) < 15:
        status = "COLLECTING DATA"
    elif s40["win"] >= 0.75 and s40["avg"] >= 0:
        status = "ON TRACK"
    elif s40["win"] >= 0.65 and s40["avg"] >= -0.005:
        status = "WATCH"
    else:
        status = "ALERT"
    report["accuracy"] = dict(last40=s40, last13w=tstats(closed[closed[f"k_{cur}"] >= asof - pd.Timedelta(weeks=13)][f"r_{cur}"]),
                              expectation=EXPECT, status=status)
    if status == "ALERT" and not lp.get("paused"):
        lp["paused"] = True
        report["actions"].append("Paused longs: the last 40 trades fell well below the backtest (ALERT).")
    elif status == "ON TRACK" and lp.get("paused"):
        lp["paused"] = False
        report["actions"].append("Resumed longs: the last 40 trades are back on track.")

    # 2) guarded re-tune over the last 26 weeks
    lo = asof - pd.Timedelta(weeks=26)
    cands = []
    for m, H in GRID:
        h = pk[(pk.date >= lo) & (pk[f"k_{m}_{H}"] <= asof)][f"r_{m}_{H}"].dropna()
        if len(h) >= 30:
            cands.append(dict(target_atr=m, max_days=H, n=len(h), win=round(float((h > 0).mean()), 3), avg=round(float(h.mean()), 5)))
    report["retune_table"] = cands
    ok = [c for c in cands if c["win"] >= 0.80]
    if ok:
        best = max(ok, key=lambda c: c["avg"])
        if (best["target_atr"], best["max_days"]) != (lp["target_atr"], lp["max_days"]):
            report["actions"].append(f"Re-tuned longs: target {lp['target_atr']}→{best['target_atr']}×ATR, "
                                     f"hold {lp['max_days']}→{best['max_days']} days (last 26 weeks: {best['win']:.0%} profitable, "
                                     f"avg {best['avg']*100:+.2f}%/trade).")
            lp["target_atr"], lp["max_days"] = best["target_atr"], best["max_days"]

    # 3) short re-test (last 52 weeks)
    sp = params["short"]
    spk, _ = rule_outcomes("short", sp["rank_by"], [(sp["target_atr"], sp["max_days"])])
    key = f"{sp['target_atr']}_{sp['max_days']}"
    sh = spk[(spk.date >= asof - pd.Timedelta(weeks=52)) & (spk[f"k_{key}"] <= asof)][f"r_{key}"]
    ss = tstats(sh)
    passes = ss.get("n", 0) >= 100 and ss["avg"] >= 0.005 and ss["win"] >= 0.80 and ss["worst"] >= -0.35
    report["short_retest"] = dict(stats=ss, passes=bool(passes),
                                  bar="avg ≥ +0.5%, ≥80% profitable, no loss worse than −35%, ≥100 trades")
    if passes != sp["enabled"]:
        sp["enabled"] = bool(passes)
        report["actions"].append(("Switched shorts ON" if passes else "Switched shorts OFF") + " after the 52-week re-test.")

    # 4) mistake review (last 13 weeks of the current rule)
    feats = pd.read_pickle(f"{ROOT}/data/features.pkl")
    rec = closed[closed[f"k_{cur}"] >= asof - pd.Timedelta(weeks=13)][["date", "symbol", f"r_{cur}"]]
    rec = rec.merge(feats[["date", "symbol"] + list(DESCRIBE)], on=["date", "symbol"], how="left")
    lose, win = rec[rec[f"r_{cur}"] <= 0], rec[rec[f"r_{cur}"] > 0]
    notes = []
    if len(lose) >= 3 and len(win) >= 3:
        for col, name in DESCRIBE.items():
            sd = rec[col].std()
            if not sd or not np.isfinite(sd):
                continue
            notes.append(dict(key=col, feature=name, losers=float(lose[col].mean()), winners=float(win[col].mean()),
                              gap=float((lose[col].mean() - win[col].mean()) / sd)))
        notes = sorted(notes, key=lambda n: -abs(n["gap"]))[:3]
    report["mistakes"] = dict(n_losers=int(len(lose)), n_winners=int(len(win)), patterns=notes,
                              worst=rec.nsmallest(3, f"r_{cur}").assign(date=lambda x: x.date.dt.date.astype(str))
                              [["date", "symbol", f"r_{cur}"]].rename(columns={f"r_{cur}": "ret"}).to_dict("records"))

    # save
    if params != old:
        params["version"] = old.get("version", 1) + 1
        params["updated"] = str(today.date())
        json.dump(params, open(PARAMS_PATH, "w"), indent=2)
        with open(f"{ROOT}/model/changelog.md", "a") as fh:
            fh.write(f"- {today.date()} · v{params['version']} · " + " ".join(report["actions"]) + "\n")
    report["params_after"] = params
    os.makedirs(f"{ROOT}/out/weekly", exist_ok=True)
    json.dump(report, open(f"{ROOT}/out/weekly/{today.date()}.json", "w"), indent=1, default=str)
    print(json.dumps({k: report[k] for k in ["accuracy", "actions", "short_retest"]}, indent=1, default=str))
    return report


if __name__ == "__main__":
    main()
