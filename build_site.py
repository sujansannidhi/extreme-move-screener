"""Build the static site in public/ (served by Vercel).

  public/index.html     next orders with dates (longs, shorts), trade log with SUCCESS/FAIL tags,
                        weekly model check, backtest
  public/research.html  the full research report (copy of report.html)
"""
import glob
import html
import json
import os

import numpy as np
import pandas as pd

from strategy import ROOT, load_params
from ledger import build_ledger

E = lambda s: html.escape(str(s))


def pct(x, d=1, signed=True):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return (f"{x*100:+.{d}f}%" if signed else f"{x*100:.{d}f}%").replace("-", "−")


def money(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"${x:,.2f}" if np.isfinite(x) else "—"


def nice(d):
    try:
        return pd.Timestamp(d).strftime("%a %b %-d")
    except Exception:
        return "—"


TAG = {"SUCCESS": ("ok", "Success"), "FAIL": ("bad", "Fail"), "OPEN": ("open", "Open"), "PENDING": ("pend", "Pending"),
       "WATCH:SUCCESS": ("ok", "Would have worked"), "WATCH:FAIL": ("bad", "Would have failed"),
       "WATCH:OPEN": ("open", "Watching"), "WATCH": ("pend", "Watch")}


def tag_chip(t):
    c, label = TAG.get(t, ("pend", t))
    return f"<span class='chip {c}'>{E(label)}</span>"


def log_rows(df):
    out = []
    for r in df.itertuples():
        cls = "pos" if np.isfinite(r.ret) and r.ret > 0 else ("neg" if np.isfinite(r.ret) and r.ret < 0 else "")
        out.append(f"<tr data-tag='{E(str(r.tag).replace('WATCH:', ''))}'><td>{tag_chip(r.tag)}</td><td>{nice(r.order_date)}</td>"
                   f"<th>{E(r.symbol)}</th><td>{money(r.entry)}</td><td>{money(r.target)}</td><td>{money(r.exit)}</td>"
                   f"<td class='{cls}'>{pct(r.ret, 2)}</td><td>{E(r.exit_reason or '—')}</td><td>{nice(r.exit_date) if r.exit_date else '—'}</td></tr>")
    return "".join(out)


def summary(df):
    c = df[df.tag.isin(["SUCCESS", "FAIL"])]
    if c.empty:
        return "No closed trades yet."
    return (f"{len(c)} closed: {(c.tag == 'SUCCESS').sum()} success, {(c.tag == 'FAIL').sum()} fail "
            f"({(c.tag == 'SUCCESS').mean():.0%}), average {pct(c.ret.mean(), 2)}, worst {pct(c.ret.min(), 1)}.")


def main():
    os.makedirs(f"{ROOT}/public", exist_ok=True)
    params = load_params()
    led = build_ledger()
    plans = sorted(glob.glob(f"{ROOT}/out/plan_*.csv"))
    latest = pd.read_csv(plans[-1]) if plans else pd.DataFrame(columns=["side"])
    asof = pd.read_csv(f"{ROOT}/data/ohlcv_full.csv.gz", usecols=["date"]).date.max()
    weekly = sorted(glob.glob(f"{ROOT}/out/weekly/*.json"))
    wk = json.load(open(weekly[-1])) if weekly else None

    # ----- next orders
    longs = latest[latest.side.isin(["LONG", "LONG_PAUSED"])]
    shorts = latest[latest.side.isin(["SHORT", "SHORT_WATCH"])]
    order_day = nice(latest.order_date.iloc[0]) if len(latest) else "—"
    sell_by = nice(longs.sell_by.iloc[0]) if len(longs) else "—"

    def order_rows(df, verb):
        return "".join(
            f"<tr><th>{E(r.symbol)}</th><td>{money(r.last_close)}</td><td class='{'pos' if r.target_pct > 0 else 'neg'}'>{pct(r.target_pct, 2)}</td>"
            f"<td><b>{money(r.target_if_fill_at_last_close)}</b></td><td>{verb} MOC · {nice(r.order_date)}</td><td>{nice(r.sell_by)} close</td>"
            f"<td>{pct(r.ret_5)}</td><td>{pct(r.ret_60, 0)}</td><td>{'' if pd.isna(r.rev_score) else f'{r.rev_score:.0f}'}</td></tr>"
            for r in df.itertuples()) or "<tr><td colspan='9'>No extreme movers qualified. No orders.</td></tr>"

    long_paused = params["long"].get("paused")
    short_on = params["short"]["enabled"] and not params["short"].get("paused")
    hdr = ("<tr><th>Ticker</th><th>Last close</th><th>Target</th><th>Exit price*</th><th>Order</th><th>Exit by</th>"
           "<th>5D</th><th>60D</th><th title='Reversal Score: higher means more pullback risk'>Rev.</th></tr>")

    # ----- trade log
    lg = led[led.side.isin(["LONG", "LONG_PAUSED"])].sort_values(["order_date", "symbol"], ascending=[False, True])
    live_l, sim_l = lg[lg.source == "LIVE"], lg[lg.source == "SIMULATED"]
    sw = led[led.side.isin(["SHORT", "SHORT_WATCH"])].sort_values(["order_date", "symbol"], ascending=[False, True])

    # ----- weekly card
    if wk:
        a = wk["accuracy"]
        st = a["status"]
        stc = {"ON TRACK": "ok", "WATCH": "open", "ALERT": "bad"}.get(st, "pend")
        l40 = a["last40"]
        acts = "".join(f"<li>{E(x)}</li>" for x in wk["actions"]) or "<li>No changes this week.</li>"
        rt = "".join(f"<tr><td>{c['target_atr']}×ATR</td><td>{c['max_days']} days</td><td>{c['n']}</td><td>{pct(c['win'], 0, False)}</td>"
                     f"<td>{pct(c['avg'], 2)}</td></tr>" for c in wk["retune_table"])
        def fv(p, v):
            return pct(v, 1, False) if p.get("key", "").startswith(("ret_", "atr", "dist", "mkt")) else f"{v:.1f}"
        pats = "".join(f"<li>Losing trades had a {'higher' if p['gap'] > 0 else 'lower'} {E(p['feature'])}: "
                       f"{fv(p, p['losers'])} vs {fv(p, p['winners'])} for winners.</li>"
                       for p in wk["mistakes"]["patterns"]) or "<li>Not enough losing trades to compare.</li>"
        sr = wk["short_retest"]
        weekly_html = f"""
<div class="cards">
 <div class="card"><span class="eb">Status · week of {nice(wk['date'])}</span><div class="big"><span class="chip {stc}">{E(st)}</span></div>
  <p>Last 40 closed trades: {pct(l40.get('win'), 0, False)} profitable, average {pct(l40.get('avg'), 2)}. Backtest expectation: about 83% profitable, average +0.3% to +0.6%.</p></div>
 <div class="card"><span class="eb">Changes made</span><ul>{acts}</ul><p class="note">Current rule: target {params['long']['target_atr']}×ATR, up to {params['long']['max_days']} sessions (v{params.get('version', 1)}).</p></div>
 <div class="card"><span class="eb">Mistake review · last 13 weeks</span><ul>{pats}</ul><p class="note">{wk['mistakes']['n_losers']} losing vs {wk['mistakes']['n_winners']} winning trades. Patterns are reported, not acted on, until they hold up on new data.</p></div>
</div>
<details><summary>Re-tune table (last 26 weeks) and short re-test</summary>
<div class="scroll"><table><thead><tr><th>Target</th><th>Max hold</th><th>Trades</th><th>Profitable</th><th>Avg</th></tr></thead><tbody>{rt}</tbody></table></div>
<p class="note">The weekly check keeps the setting with the best average among those at least 80% profitable. Short re-test (52 weeks): {sr['stats'].get('n', 0)} trades, {pct(sr['stats'].get('win'), 0, False)} profitable, average {pct(sr['stats'].get('avg'), 2)}, worst {pct(sr['stats'].get('worst'), 0)} → {'passes' if sr['passes'] else 'fails'} the bar ({E(sr['bar'])}).</p></details>"""
    else:
        weekly_html = "<p>The first weekly check runs Saturday.</p>"

    changelog = open(f"{ROOT}/model/changelog.md").read().split("\n", 3)[-1] if os.path.exists(f"{ROOT}/model/changelog.md") else ""
    cl_items = "".join(f"<li>{E(l[2:])}</li>" for l in changelog.splitlines() if l.startswith("- "))

    bt = ""
    for k, lab in [("dev", "Development · Jul 2023 – Jun 2025"), ("holdout", "Holdout · Jul 2025 – Oct 2026")]:
        p = f"{ROOT}/out/nextday_{k}.json"
        if os.path.exists(p):
            j = json.load(open(p))
            t, pf = j["trades"], j["portfolio"]
            bt += (f"<tr><th>{lab}</th><td>{t['n']:,}</td><td>{pct(t['win'], 0, False)}</td><td>{pct(t['avg'], 2)}</td>"
                   f"<td class='neg'>{pct(t['p05'])}</td><td>{pct(pf['total'], 0)}</td><td class='neg'>{pct(pf['mdd'], 0, False)}</td></tr>")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Extreme Move Screener</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@700;800&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#f4f6f7;--panel:#fff;--ink:#15212b;--ink2:#4a5966;--rule:#d5dde3;--tint:#e8eef2;--accent:#0b5c7a;--gain:#11734b;--loss:#b3261e;--warn:#8a5a00;
 --okbg:#dff1e8;--badbg:#fbe3e1;--openbg:#fff1d6;--pendbg:#e8eef2;
 --disp:"Archivo","Helvetica Neue",Arial,sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root{{--paper:#10171d;--panel:#16212a;--ink:#e3eaef;--ink2:#9fb0bd;--rule:#2a3843;--tint:#1c2933;--accent:#5fb6d6;--gain:#4cc78f;--loss:#f07a70;--warn:#e0b252;
 --okbg:#173a2b;--badbg:#43211f;--openbg:#3d3115;--pendbg:#222d36;color-scheme:dark}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 var(--body)}}
.wrap{{max-width:1080px;margin:0 auto;padding:28px 18px 60px;display:grid;gap:36px}}
header{{border-bottom:2px solid var(--ink);padding-bottom:16px;display:grid;gap:8px}}
.eb{{font:12px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink2)}}
h1{{font:800 clamp(30px,5vw,44px)/1.05 var(--disp);margin:0}} h2{{font:700 22px var(--disp);margin:0}} h3{{font:700 16px var(--disp);margin:0}}
p{{margin:0;max-width:74ch}} .note{{font-size:13px;color:var(--ink2)}} section{{display:grid;gap:12px}} ul{{margin:0;padding-left:18px;display:grid;gap:4px}}
.when{{background:var(--panel);border:1px solid var(--rule);border-left:4px solid var(--accent);padding:12px 16px;display:grid;gap:4px}}
.when b{{font-family:var(--disp);font-size:18px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}}
.card{{background:var(--panel);border:1px solid var(--rule);padding:12px 14px;display:grid;gap:8px;align-content:start}} .big{{font-size:18px}}
.scroll{{overflow-x:auto;border:1px solid var(--rule);background:var(--panel)}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}}
thead th{{font:500 11px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);background:var(--tint);position:sticky;top:0}}
tbody th{{font-family:var(--mono)}} .pos{{color:var(--gain)}} .neg{{color:var(--loss)}} a{{color:var(--accent)}}
.chip{{display:inline-block;font:500 11.5px var(--mono);padding:2px 8px;border-radius:3px;white-space:nowrap}}
.chip.ok{{background:var(--okbg);color:var(--gain)}} .chip.bad{{background:var(--badbg);color:var(--loss)}}
.chip.open{{background:var(--openbg);color:var(--warn)}} .chip.pend{{background:var(--pendbg);color:var(--ink2)}}
.off{{background:var(--panel);border:1px dashed var(--rule);padding:12px 14px;display:grid;gap:6px}}
.tabs{{display:flex;gap:6px;flex-wrap:wrap}} .tabs button{{font:500 12px var(--mono);padding:5px 10px;border:1px solid var(--rule);background:var(--panel);color:var(--ink);cursor:pointer;border-radius:3px}}
.tabs button[aria-pressed=true]{{background:var(--ink);color:var(--paper)}} .log{{max-height:520px;overflow:auto}}
details summary{{cursor:pointer;font-weight:600;margin:6px 0}}
</style></head><body><div class="wrap">
<header><div class="eb">Data through {nice(asof)} close · updates every weekday after the close · model check every Saturday</div>
<h1>Extreme Move Screener</h1>
<p>Buy three of the day's most liquid extreme movers at the closing price and sell into the first small pop. Shorts are listed separately and only trade when the weekly re-test says they're safe.</p></header>

<section><h2>Next orders</h2>
<div class="when"><span class="eb">When</span><b>Place market-on-close orders on {order_day}, before 3:50pm ET</b>
<span class="note">Exit at the target as soon as it trades from the next session on (a limit order at the target does this). If it isn't hit, exit at the close on the exit-by date.</span></div>
<h3>Longs</h3>
{"<div class='off'><b>Longs are paused by the weekly check.</b><span class='note'>The list below is for watching only.</span></div>" if long_paused else ""}
<div class="scroll"><table><thead>{hdr}</thead><tbody>{order_rows(longs, "Buy")}</tbody></table></div>
<p class="note">*If you fill at the last close. Recompute from your fill: long exit = fill × (1 + target). No stop: in testing, every stop lowered the average result.</p>
<h3>Shorts</h3>
{"" if short_on else "<div class='off'><b>Shorts are off.</b><span class='note'>The best short rule averaged about +0.1% to +0.5% per trade, but single trades lost up to 89–223% when a stock kept running. Every protective stop turned it negative. The weekly re-test switches shorts on only if they clear the bar. Watch list below: the biggest 5-day gainers and where a short would have covered.</span></div>"}
<div class="scroll"><table><thead>{hdr}</thead><tbody>{order_rows(shorts, "Short" if short_on else "Watch")}</tbody></table></div>
</section>

<section><h2>Trade log</h2>
<p><b>Live:</b> {summary(live_l)} <b>Simulated (60 sessions before launch):</b> {summary(sim_l)}</p>
<div class="tabs" role="group" aria-label="Filter trade log"><button aria-pressed="true" data-f="ALL">All</button><button aria-pressed="false" data-f="SUCCESS">Success</button><button aria-pressed="false" data-f="FAIL">Fail</button><button aria-pressed="false" data-f="OPEN">Open</button><button aria-pressed="false" data-f="PENDING">Pending</button></div>
<h3>Live (published before the trade)</h3>
<div class="scroll log"><table class="flt"><thead><tr><th>Result</th><th>Bought</th><th>Ticker</th><th>Entry</th><th>Target</th><th>Exit / last</th><th>Return</th><th>What happened</th><th>Exit day</th></tr></thead><tbody>{log_rows(live_l) or "<tr><td colspan='9'>Live trades start with the first order day.</td></tr>"}</tbody></table></div>
<h3>Simulated history (before launch)</h3>
<div class="scroll log"><table class="flt"><thead><tr><th>Result</th><th>Bought</th><th>Ticker</th><th>Entry</th><th>Target</th><th>Exit / last</th><th>Return</th><th>What happened</th><th>Exit day</th></tr></thead><tbody>{log_rows(sim_l)}</tbody></table></div>
<details><summary>Short watch list history</summary><div class="scroll log"><table class="flt"><thead><tr><th>Result</th><th>Day</th><th>Ticker</th><th>Price</th><th>Target</th><th>Exit / last</th><th>Short return</th><th>What happened</th><th>Exit day</th></tr></thead><tbody>{log_rows(sw)}</tbody></table></div></details>
<p class="note">Replayed from daily bars: a target counts as hit if the day's range reaches it. Returns include selling costs (and borrow for shorts).</p>
</section>

<section><h2>Weekly model check</h2>{weekly_html}
<details><summary>Model changelog</summary><ul>{cl_items}</ul></details></section>

<section><h2>Backtest</h2>
<div class="scroll"><table><thead><tr><th>Period</th><th>Trades</th><th>Profitable</th><th>Avg trade</th><th>Worst 5%</th><th>Return</th><th>Max drawdown</th></tr></thead><tbody>{bt}</tbody></table></div>
<p class="note">About 1 in 6 trades loses, and the worst lost 25–47% in five days. Drawdowns were much deeper than the S&amp;P 500's. Weekly re-tuning was itself backtested: unguarded re-tuning was unstable, and the 80%-win guard used here matched the fixed rule. This is a research project, not financial advice. <a href="research.html">Full research report</a>.</p></section>
</div>
<script>
document.querySelectorAll('.tabs button').forEach(b=>b.addEventListener('click',()=>{{
  document.querySelectorAll('.tabs button').forEach(x=>x.setAttribute('aria-pressed',x===b));
  const f=b.dataset.f;
  document.querySelectorAll('table.flt tbody tr').forEach(tr=>{{tr.hidden = f!=='ALL' && tr.dataset.tag!==f;}});
}}));
</script></body></html>"""
    open(f"{ROOT}/public/index.html", "w").write(page)
    if os.path.exists(f"{ROOT}/report.html"):
        src = open(f"{ROOT}/report.html").read()
        if not src.lstrip().lower().startswith("<!doctype"):
            src = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>' + src + "</body></html>"
        open(f"{ROOT}/public/research.html", "w").write(src)
    print("built public/index.html", "| ledger rows:", len(led))


if __name__ == "__main__":
    main()
