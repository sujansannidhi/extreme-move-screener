"""Build the static site in public/ (served by Vercel).

  public/index.html     next buys, sell prices, paper-trade record, backtest summary
  public/research.html  the full research report (copy of report.html)

Paper-trade record: every out/plan_<date>.csv is replayed against the price data with the live rule
(buy at the next session's close; sell at entry*(1+0.25*ATR%) or at the open if it gaps above; else at the
5th session's close). Picks still inside their window are marked open and valued at the last close.
"""
import glob
import html
import json
import os
import shutil

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET_ATR, MAX_DAYS, SELL_COST = 0.25, 5, 0.0005
E = lambda s: html.escape(str(s))


def pct(x, d=1, signed=True):
    if x is None or not np.isfinite(x):
        return "—"
    return (f"{x*100:+.{d}f}%" if signed else f"{x*100:.{d}f}%").replace("-", "−")


def replay(plans, px):
    rows = []
    dates = px.index
    for p in plans:
        asof = pd.Timestamp(os.path.basename(p)[5:15])
        plan = pd.read_csv(p)
        after = dates[dates > asof]
        for r in plan.itertuples():
            row = dict(signal=asof.date(), symbol=r.symbol, status="waiting for buy day", entry=np.nan, exit=np.nan,
                       ret=np.nan, buy_day=r.buy_on_close_of, exit_day="")
            if len(after) == 0 or r.symbol not in px.columns.get_level_values(1):
                rows.append(row)
                continue
            b = after[0]
            entry = px.loc[b, ("close", r.symbol)]
            if not np.isfinite(entry):
                rows.append(row)
                continue
            tgt = entry * (1 + TARGET_ATR * r.atr_pct)
            row.update(entry=entry, buy_day=b.date(), target=tgt)
            fut = dates[dates > b][:MAX_DAYS]
            done = False
            for k, d in enumerate(fut, 1):
                o, h, c = (px.loc[d, (f, r.symbol)] for f in ("open", "high", "close"))
                if not np.isfinite(c):
                    continue
                if o >= tgt:
                    row.update(exit=o, exit_day=d.date(), status="sold at open (gap above target)"); done = True
                elif h >= tgt:
                    row.update(exit=tgt, exit_day=d.date(), status=f"target hit day {k}"); done = True
                elif k == MAX_DAYS:
                    row.update(exit=c, exit_day=d.date(), status="time exit day 5"); done = True
                if done:
                    break
            if not done:
                last = px[("close", r.symbol)].dropna()
                row.update(exit=last.iloc[-1], status=f"open · day {len(fut)} of {MAX_DAYS}")
            row["ret"] = row["exit"] / entry - 1 - (SELL_COST if done else 0)
            row["closed"] = done
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(f"{ROOT}/public", exist_ok=True)
    d = pd.read_csv(f"{ROOT}/data/ohlcv_full.csv.gz", parse_dates=["date"])
    px = d.pivot(index="date", columns="symbol", values=["open", "high", "close"]).sort_index()
    asof = d.date.max()
    plans = sorted(glob.glob(f"{ROOT}/out/plan_*.csv"))
    latest = pd.read_csv(plans[-1]) if plans else pd.DataFrame()
    rec = replay(plans, px) if plans else pd.DataFrame()

    nd = {k: json.load(open(f"{ROOT}/out/nextday_{k}.json")) for k in ["dev", "holdout"] if os.path.exists(f"{ROOT}/out/nextday_{k}.json")}

    buy_rows = "".join(
        f"<tr><th>{E(r.symbol)}</th><td>${r.price:,.2f}</td><td class='pos'>+{r.target_pct*100:.2f}%</td>"
        f"<td><b>${r.sell_price_if_fill_at_last_close:,.2f}</b></td><td>{E(r.buy_on_close_of)}</td><td>{E(r.sell_by_close_of)}</td>"
        f"<td class='{'pos' if r.ret_20 > 0 else 'neg'}'>{pct(r.ret_20)}</td><td>{pct(r.ret_60, 0)}</td></tr>"
        for r in latest.itertuples()) or "<tr><td colspan='8'>No extreme movers qualified today. No buys.</td></tr>"

    rec_rows, summary = "", "No paper trades yet. The record starts with the first plan's buy day."
    if len(rec):
        rr = rec.sort_values(["signal", "symbol"], ascending=[False, True])
        for r in rr.itertuples():
            cls = "pos" if np.isfinite(r.ret) and r.ret > 0 else ("neg" if np.isfinite(r.ret) and r.ret < 0 else "")
            rec_rows += (f"<tr><td>{E(r.buy_day)}</td><th>{E(r.symbol)}</th><td>{'$%.2f' % r.entry if np.isfinite(r.entry) else '—'}</td>"
                         f"<td>{'$%.2f' % r.exit if np.isfinite(r.exit) else '—'}</td><td class='{cls}'>{pct(r.ret, 2)}</td>"
                         f"<td>{E(r.status)}</td><td>{E(r.exit_day)}</td></tr>")
        cl = rec[rec.get("closed", False) == True] if "closed" in rec else rec.iloc[0:0]
        if len(cl):
            summary = (f"{len(cl)} closed paper trades: {(cl.ret > 0).mean()*100:.0f}% profitable, average {pct(cl.ret.mean(), 2)}, "
                       f"worst {pct(cl.ret.min(), 1)}. Backtest expectation: about 83% profitable, average +0.3% to +0.6%.")
        else:
            summary = "Trades are open or waiting for their buy day. Results appear here as they close."

    bt_rows = ""
    for k, lab in [("dev", "Development · Jul 2023 – Jun 2025"), ("holdout", "Holdout · Jul 2025 – Oct 2026")]:
        if k in nd:
            t, p = nd[k]["trades"], nd[k]["portfolio"]
            bt_rows += (f"<tr><th>{lab}</th><td>{t['n']:,}</td><td>{pct(t['win'], 0, False)}</td><td>{pct(t['avg'], 2)}</td>"
                        f"<td class='neg'>{pct(t['p05'])}</td><td>{pct(p['total'], 0)}</td><td class='neg'>{pct(p['mdd'], 0, False)}</td></tr>")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Extreme Move Screener</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@700;800&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#f4f6f7;--panel:#fff;--ink:#15212b;--ink2:#4a5966;--rule:#d5dde3;--tint:#e8eef2;--accent:#0b5c7a;--gain:#11734b;--loss:#b3261e;
 --disp:"Archivo","Helvetica Neue",Arial,sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root{{--paper:#10171d;--panel:#16212a;--ink:#e3eaef;--ink2:#9fb0bd;--rule:#2a3843;--tint:#1c2933;--accent:#5fb6d6;--gain:#4cc78f;--loss:#f07a70;color-scheme:dark}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 var(--body)}}
.wrap{{max-width:1040px;margin:0 auto;padding:28px 18px 60px;display:grid;gap:34px}}
header{{border-bottom:2px solid var(--ink);padding-bottom:16px;display:grid;gap:8px}}
.eb{{font:12px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink2)}}
h1{{font:800 clamp(30px,5vw,44px)/1.05 var(--disp);margin:0}} h2{{font:700 22px var(--disp);margin:0}}
p{{margin:0;max-width:72ch}} .note{{font-size:13px;color:var(--ink2)}} section{{display:grid;gap:12px}}
.steps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.steps div{{background:var(--panel);border:1px solid var(--rule);padding:12px 14px;display:grid;gap:4px}}
.scroll{{overflow-x:auto;border:1px solid var(--rule);background:var(--panel)}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}}
thead th{{font:500 11px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);background:var(--tint)}}
tbody th{{font-family:var(--mono)}} .pos{{color:var(--gain)}} .neg{{color:var(--loss)}} a{{color:var(--accent)}}
</style></head><body><div class="wrap">
<header><div class="eb">Data through {asof:%a %b %d, %Y} close · updated automatically every weekday</div>
<h1>Extreme Move Screener</h1>
<p>Buy three of the day's most liquid extreme movers at the closing price, then sell into the first small pop. The rule was fixed before its out-of-sample test and is tracked here with paper trades.</p></header>
<section><h2>Next buys</h2>
<div class="steps"><div><span class="eb">Buy</span><p>Market-on-close order on the buy day. Fills at that day's close. Split capital three ways.</p></div>
<div><span class="eb">Sell</span><p>Limit sell at your fill price plus the target %. If it opens above the target, sell at the open.</p></div>
<div><span class="eb">Time limit</span><p>Not hit by the sell-by date? Sell at that day's close. No stop.</p></div></div>
<div class="scroll"><table><thead><tr><th>Ticker</th><th>Last close</th><th>Target</th><th>Sell price*</th><th>Buy at close of</th><th>Sell by close of</th><th>20D</th><th>60D</th></tr></thead><tbody>{buy_rows}</tbody></table></div>
<p class="note">*If you fill at the last close. Recompute from your actual fill price. Exchange holidays are not modeled in the dates.</p></section>
<section><h2>Paper-trade record</h2><p>{E(summary)}</p>
<div class="scroll"><table><thead><tr><th>Bought</th><th>Ticker</th><th>Entry</th><th>Exit / last</th><th>Result</th><th>Status</th><th>Exit day</th></tr></thead><tbody>{rec_rows or "<tr><td colspan='7'>—</td></tr>"}</tbody></table></div>
<p class="note">Replayed from daily bars: the target counts as hit if the day's high reaches it. Results include 5 bps selling cost.</p></section>
<section><h2>Backtest</h2>
<div class="scroll"><table><thead><tr><th>Period</th><th>Trades</th><th>Profitable</th><th>Avg trade</th><th>Worst 5%</th><th>Return</th><th>Max drawdown</th></tr></thead><tbody>{bt_rows}</tbody></table></div>
<p class="note">About 1 in 6 trades loses, and the worst lost 25–47% in five days. Drawdowns were much deeper than the S&amp;P 500's. This is a research project, not financial advice. <a href="research.html">Full research report</a>.</p></section>
</div></body></html>"""
    open(f"{ROOT}/public/index.html", "w").write(page)
    if os.path.exists(f"{ROOT}/report.html"):
        src = open(f"{ROOT}/report.html").read()
        if not src.lstrip().lower().startswith("<!doctype"):
            src = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>' + src + "</body></html>"
        open(f"{ROOT}/public/research.html", "w").write(src)
    print("built public/index.html as of", asof.date(), "| plans:", len(plans), "| paper trades:", len(rec))


if __name__ == "__main__":
    main()
