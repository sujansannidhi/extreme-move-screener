"""Build the single-page HTML report from backtest results, today's screen and catalyst notes."""
import json, glob, html
import numpy as np
import pandas as pd
from features import ROOT, load_panels

R = json.load(open(f"{ROOT}/out/backtest_results.json"))
S = pd.read_csv(f"{ROOT}/out/screen_2026-10-02.csv")
S = S[S.candidate & (S.symbol != "SOXL")].copy()
CAT = {json.load(open(f))["symbol"]: json.load(open(f)) for f in glob.glob(f"{ROOT}/data/catalysts/*.json")}
feats = pd.read_pickle(f"{ROOT}/data/features.pkl")
from backtest import classify
t = feats[(feats.date == feats.date.max()) & feats.symbol.isin(S.symbol)].copy()
reg, _ = classify(t, "rules")
RULES = dict(zip(t.symbol, reg))
E = lambda s: html.escape(str(s))


def pct(x, signed=True, d=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x*100:+.{d}f}%" if signed else f"{x*100:.{d}f}%"
    return s.replace("-", "−")


def num(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{d}f}".replace("-", "−")


def cls(x):
    return "pos" if x is not None and np.isfinite(x) and x > 0 else ("neg" if x is not None and np.isfinite(x) and x < 0 else "")


# ------------------------------------------------------------------ equity chart
P = load_panels()
spy = P["close"]["SPY"]
ef = pd.read_csv(f"{ROOT}/out/equity_final_2023-07.csv", parse_dates=["date"]).set_index("date").equity
er = pd.read_csv(f"{ROOT}/out/equity_rules_2023-07.csv", parse_dates=["date"]).set_index("date").equity
idx = ef.index.union(er.index)
ser = {"Rules (untuned)": er.reindex(idx).ffill(), "Final model": ef.reindex(idx).ffill(),
       "SPY": spy.reindex(idx).ffill()}
ser = {k: v / v.dropna().iloc[0] for k, v in ser.items()}
W, H, L, Rm, T, B = 880, 300, 46, 16, 14, 30
ymin = 0.8
ymax = max(v.max() for v in ser.values()) * 1.04
n = len(idx)
X = lambda i: L + (W - L - Rm) * i / (n - 1)
Y = lambda v: T + (H - T - B) * (1 - (v - ymin) / (ymax - ymin))
ho = int(np.searchsorted(idx, pd.Timestamp("2025-07-01")))
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Equity curves, out-of-sample July 2023 to October 2026">']
svg.append(f'<rect x="{X(ho):.1f}" y="{T}" width="{X(n-1)-X(ho):.1f}" height="{H-T-B}" class="ho"/>')
svg.append(f'<text x="{X(ho)+6:.1f}" y="{T+14}" class="lbl">Holdout (not tuned on)</text>')
for v in np.arange(1.0, ymax, 0.5):
    svg.append(f'<line x1="{L}" x2="{W-Rm}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
    svg.append(f'<text x="{L-6}" y="{Y(v)+4:.1f}" class="ax" text-anchor="end">{v:.1f}×</text>')
for y in [2024, 2025, 2026]:
    i = int(np.searchsorted(idx, pd.Timestamp(f"{y}-01-01")))
    svg.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{H-B}" y2="{H-B+5}" class="axl"/>')
    svg.append(f'<text x="{X(i):.1f}" y="{H-B+18}" class="ax" text-anchor="middle">{y}</text>')
svg.append(f'<text x="{L}" y="{H-B+18}" class="ax" text-anchor="start">Jul 2023</text>')
for k, c in [("SPY", "s-spy"), ("Rules (untuned)", "s-rules"), ("Final model", "s-final")]:
    v = ser[k].values
    pts = " ".join(f"{X(i):.1f},{Y(x):.1f}" for i, x in enumerate(v) if np.isfinite(x))
    svg.append(f'<polyline points="{pts}" class="{c}"/>')
    svg.append(f'<circle cx="{X(n-1):.1f}" cy="{Y(v[-1]):.1f}" r="3" class="{c}-dot"/>')
svg.append("</svg>")
legend = "".join(f'<span class="lg"><i class="{c}-k"></i>{k} <b>{pct(ser[k].iloc[-1]-1, d=0)}</b></span>'
                 for k, c in [("Rules (untuned)", "s-rules"), ("Final model", "s-final"), ("SPY", "s-spy")])

# ------------------------------------------------------------------ backtest tables
def strat_rows(per):
    b = R[per]
    rows = []
    names = {"final": "Final model (pre-registered)", "rules": "Rules only (untuned)", "icw": "IC-weighted composite", "ml": "Gradient boosting"}
    for m in ["final", "rules", "icw", "ml"]:
        e = b[m]["equity"]
        rows.append(f"<tr><th>{names[m]}</th><td class='{cls(e['total_return'])}'>{pct(e['total_return'],d=0)}</td>"
                    f"<td>{num(e['sharpe'])}</td><td class='neg'>{pct(e['max_drawdown'],False,0)}</td>"
                    f"<td>{pct(e['pct_3m_windows_positive'],False,0)}</td><td class='{cls(e['median_3m_return'])}'>{pct(e['median_3m_return'])}</td>"
                    f"<td class='neg'>{pct(e['worst_3m_return'],d=0)}</td></tr>")
    e = b["SPY"]
    rows.append(f"<tr class='bench'><th>SPY buy &amp; hold</th><td class='{cls(e['total_return'])}'>{pct(e['total_return'],d=0)}</td>"
                f"<td>{num(e['sharpe'])}</td><td class='neg'>{pct(e['max_drawdown'],False,0)}</td>"
                f"<td>{pct(e['pct_3m_windows_positive'],False,0)}</td><td class='{cls(e['median_3m_return'])}'>{pct(e['median_3m_return'])}</td>"
                f"<td class='neg'>{pct(e['worst_3m_return'],d=0)}</td></tr>")
    return "".join(rows)


def side_rows(mode):
    rows = []
    for per, lab in [("development", "Development"), ("holdout", "Holdout"), ("full", "Full OOS")]:
        for side in ["MOMENTUM", "REVERSAL"]:
            s = R[per][mode]["signals"][side]
            eo = R[per][mode].get(f"equity_{side}_only")
            if not s.get("n"):
                rows.append(f"<tr><th>{lab}</th><td>{side.title()}</td><td colspan='7' class='muted'>no signals</td></tr>")
                continue
            rows.append(f"<tr><th>{lab}</th><td><span class='pill {'p-long' if side=='MOMENTUM' else 'p-short'}'>{side.title()}</span></td>"
                        f"<td>{s['n']:,}</td><td>{pct(s['win_rate'],False,0)}</td><td class='{cls(s['avg'])}'>{pct(s['avg'])}</td>"
                        f"<td class='{cls(s['median'])}'>{pct(s['median'])}</td><td>{num(s['profit_factor'])}</td>"
                        f"<td>{num(eo['sharpe']) if eo else '—'}</td><td class='neg'>{pct(eo['max_drawdown'],False,0) if eo else '—'}</td></tr>")
    return "".join(rows)


def base_rows():
    names = {"long_all_candidates": "Buy every extreme mover (10d)", "short_all_candidates": "Short every extreme mover (5d)",
             "short_RSI80": "Short every mover with RSI ≥ 80", "short_all_confirmed_breaks": "Short movers after a confirmed break",
             "long_top_decile_ret20": "Buy top-decile 20-day strength", "long_fresh_breakout_gate_only": "Buy fresh breakouts (gate only)",
             "short_blowoff_gate": "Short blow-off pattern"}
    rows = []
    for k, lab in names.items():
        d, h = R["development"]["baselines"][k], R["holdout"]["baselines"][k]
        rows.append(f"<tr><th>{lab}</th><td>{d['n']:,}</td><td class='{cls(d['avg'])}'>{pct(d['avg'])}</td><td>{pct(d['win_rate'],False,0)}</td>"
                    f"<td>{h['n']:,}</td><td class='{cls(h['avg'])}'>{pct(h['avg'])}</td><td>{pct(h['win_rate'],False,0)}</td></tr>")
    return "".join(rows)


oos = pd.read_pickle(f"{ROOT}/out/oos.pkl")
oos["per"] = np.where(oos.date < "2025-07-01", "dev", "ho")
cal_rows = []
for col, y, lab in [("rev_prob", "y_rev", "Reversal Score"), ("mom_prob", "y_cont", "Momentum Score")]:
    x = oos.dropna(subset=[y]).copy()
    x["b"] = pd.qcut(x[col], 5, labels=False)
    g = x.groupby(["b", "per"]).agg(pred=(col, "mean"), act=(y, "mean")).unstack("per")
    cells = "".join(f"<td>{g.loc[q,('pred','dev')]*100:.0f} → <b>{g.loc[q,('act','dev')]*100:.0f}</b></td>"
                    f"<td>{g.loc[q,('pred','ho')]*100:.0f} → <b>{g.loc[q,('act','ho')]*100:.0f}</b></td>" for q in range(5))
    cal_rows.append(f"<tr><th>{lab}</th>{cells}</tr>")
auc = {p: R[p]["auc"] for p in ["development", "holdout"]}

# ------------------------------------------------------------------ opportunity tables
def cat_cell(sym, short=False):
    c = CAT.get(sym)
    if not c:
        return "<span class='muted'>not researched</span>"
    sup = c["support"]
    tag = f"<span class='tag t-{sup}'>{E(sup)}</span>"
    date = f" · {c['catalyst_date']}" if c.get("catalyst_date") else ""
    if short:
        return f"{tag} {E(c['catalyst_type'])}{E(date)}"
    nx = c.get("next_earnings_date")
    nxt = f"<div class='sub'>Next earnings {E(nx)}</div>" if nx else ""
    dil = c.get("dilution_or_offering", "")
    dl = f"<div class='sub'>Dilution: {E(dil)}</div>" if dil and not dil.lower().startswith("none found") else ""
    return f"{tag} <b>{E(c['catalyst_type'])}</b>{E(date)}<div class='sum'>{E(c['summary'])}</div>{dl}{nxt}"


def opp_row(r, full=True):
    d = r.direction if r.direction in ("LONG", "SHORT") else ("LONG*" if RULES.get(r.symbol) == "MOMENTUM" else "—")
    pill = {"LONG": "p-long", "SHORT": "p-short", "LONG*": "p-alt"}.get(d, "p-none")
    setup = r.setup.replace("No trade: ", "")
    if d == "LONG*":
        atr = r.ATR
        entry, inval, hold = "Buy at Mon 10/5 close (MOC)", f"Close < ${r.price-2*atr:,.2f} (entry − 2 ATR)", "10 sessions"
        setup = "Rules-only momentum (final model: " + setup + ")"
    else:
        entry, inval, hold = r.entry_condition, r.invalidation, r.holding_period
    return (f"<tr class='main'><th class='tk'>{E(r.symbol)}<div class='sub'>{E(r.peer_group)}</div></th>"
            f"<td><span class='pill {pill}'>{E(d)}</span></td><td class='setup'>{E(setup)}<div class='sub'>{E(r.move_type)} move</div></td>"
            f"<td class='sc'>{r.EMS:.0f}</td><td class='sc'>{r.MomentumScore:.0f}</td><td class='sc'>{r.ReversalScore:.0f}</td>"
            f"<td class='{cls(r.ret_1)}'>{pct(r.ret_1)}</td><td class='{cls(r.ret_5)}'>{pct(r.ret_5)}</td>"
            f"<td class='{cls(r.ret_20)}'>{pct(r.ret_20)}</td><td class='{cls(r.ret_60)}'>{pct(r.ret_60,d=0)}</td>"
            f"<td>{r.rvol:.1f}×<div class='sub'>5d max {r.rvol_max5:.1f}×</div></td><td>{pct(r.rv20,False,0)}<div class='sub'>ATR {pct(r.atr_pct,False)}</div></td>"
            f"<td class='lv'>{E(entry)}</td><td class='lv'>{E(inval)}</td><td>{E(hold)}</td></tr>"
            f"<tr class='detail'><td></td><td colspan='14' class='cat'><div class='inner'><span class='eyebrow'>Catalyst</span> {cat_cell(r.symbol)}</div></td></tr>")


S["rank_key"] = np.where(S.direction != "—", 0, np.where(S.symbol.map(RULES) == "MOMENTUM", 1, 2))
S = S.sort_values(["rank_key", "MomentumScore", "EMS"], ascending=[True, False, False])
act = S[S.rank_key < 2]
watch = S[S.rank_key == 2].sort_values("EMS", ascending=False)
act_rows = "".join(opp_row(r) for r in act.itertuples())
watch_rows = "".join(opp_row(r) for r in watch.itertuples())

thead = ("<tr><th>Ticker</th><th>Direction</th><th>Setup</th><th title='Extreme Move Score'>Extreme</th><th title='Momentum Continuation Score: walk-forward probability the long trade nets more than 3%'>Momentum</th>"
         "<th title='Reversal Score: walk-forward probability a 5-day short nets more than 3%'>Reversal</th><th>1D</th><th>5D</th><th>20D</th><th>60D</th><th>Rel. volume</th><th>Volatility (20d ann.)</th>"
         "<th>Entry</th><th>Invalidation (close basis)</th><th>Hold</th></tr>")

full, hold, dev = R["full"], R["holdout"], R["development"]
coefs_m = R["lr_final_coefs"]["momentum"]
coefs_r = R["lr_final_coefs"]["reversal"]


def coef_list(d, k=5):
    s = pd.Series(d).sort_values(key=abs, ascending=False).head(k)
    return ", ".join(f"<code>{E(i)}</code> {v:+.2f}".replace("-", "−") for i, v in s.items())


# ------------------------------------------------------------------ buy-at-close plan
ND = {k: json.load(open(f"{ROOT}/out/nextday_{k}.json")) for k in ["dev", "holdout"]}
from backtest import candidate_mask as _cm
_t = feats[(feats.date == feats.date.max())].copy()
_t = _t[(_t.price >= 3) & (_t.dollar_vol20 >= 15e6) & _t.symbol.isin(S.symbol)]
_t = _t[_cm(_t)].sort_values("dollar_vol20", ascending=False).head(3)
plan_rows = ""
for r in _t.itertuples():
    tp = 0.25 * r.atr_pct
    c = CAT.get(r.symbol, {})
    ev = {"MRVL": "Investor day Tue Oct 6 (inside the hold window)"}.get(r.symbol, "")
    plan_rows += (f"<tr><th class='tk'>{E(r.symbol)}</th><td>${r.price:,.2f}</td><td class='pos'>+{tp*100:.2f}%</td>"
                  f"<td><b>${r.price*(1+tp):,.2f}</b><div class='sub'>if you fill at ${r.price:,.2f}</div></td>"
                  f"<td>{pct(r.ret_20)}</td><td>{pct(r.atr_pct, False)}</td>"
                  f"<td class='cat'>{E(c.get('summary','')[:160])}{'…' if len(c.get('summary',''))>160 else ''}"
                  f"{'<div class=sub><b>'+E(ev)+'</b></div>' if ev else ''}</td></tr>")
spy_ho = P["close"]["SPY"].loc["2025-07-01":]
spy_dev = P["close"]["SPY"].loc["2023-07-01":"2025-06-30"]
def _spy(sr):
    r = sr.pct_change().dropna(); dd = sr / sr.cummax() - 1
    return dict(total=sr.iloc[-1] / sr.iloc[0] - 1, sharpe=r.mean() / r.std() * np.sqrt(252), mdd=dd.min())
SPYS = {"dev": _spy(spy_dev), "holdout": _spy(spy_ho)}
def plan_stat_rows():
    out = ""
    lab = {"dev": "Development · Jul 2023 – Jun 2025", "holdout": "Holdout · Jul 2025 – Oct 2026 (run once)"}
    for k in ["dev", "holdout"]:
        t, p = ND[k]["trades"], ND[k]["portfolio"]
        out += (f"<tr><th>{lab[k]}</th><td>{t['n']:,}</td><td>{pct(t['win'],False,0)}</td><td>{pct(t['exit_with_profit_by_day']['1'],False,0)}</td>"
                f"<td>{pct(t['green_at_day1_close'],False,0)}</td><td class='{cls(t['avg'])}'>{pct(t['avg'],d=2)}</td><td class='neg'>{pct(t['p05'],d=1)}</td>"
                f"<td>{num(t['pf'])}</td><td class='{cls(p['total'])}'>{pct(p['total'],d=0)}<div class='sub'>SPY {pct(SPYS[k]['total'],d=0)}</div></td>"
                f"<td>{num(p['sharpe'])}<div class='sub'>SPY {num(SPYS[k]['sharpe'])}</div></td>"
                f"<td class='neg'>{pct(p['mdd'],False,0)}<div class='sub'>SPY {pct(SPYS[k]['mdd'],False,0)}</div></td></tr>")
    return out

html_out = f"""<title>Extreme Mover Screen</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a desk research note. Narrow reading column for prose, full-bleed scroll areas for the wide tables. */
:root {{
  --paper:#f4f6f7; --panel:#ffffff; --ink:#15212b; --ink-2:#4a5966; --rule:#d5dde3; --tint:#e8eef2;
  --accent:#0b5c7a; --gain:#11734b; --loss:#b3261e; --warn:#8a5a00; --ho:#e9eef6;
  --long-bg:#dff1e8; --short-bg:#fbe3e1; --alt-bg:#e3edf6; --none-bg:#eceff1;
  --f-disp:"Archivo","Helvetica Neue",Arial,sans-serif; --f-body:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; --f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper:#10171d; --panel:#16212a; --ink:#e3eaef; --ink-2:#9fb0bd; --rule:#2a3843; --tint:#1c2933;
  --accent:#5fb6d6; --gain:#4cc78f; --loss:#f07a70; --warn:#e0b252; --ho:#1a2632;
  --long-bg:#173a2b; --short-bg:#43211f; --alt-bg:#1b3346; --none-bg:#222d36; color-scheme:dark; }} }}
:root[data-theme="dark"] {{
  --paper:#10171d; --panel:#16212a; --ink:#e3eaef; --ink-2:#9fb0bd; --rule:#2a3843; --tint:#1c2933;
  --accent:#5fb6d6; --gain:#4cc78f; --loss:#f07a70; --warn:#e0b252; --ho:#1a2632;
  --long-bg:#173a2b; --short-bg:#43211f; --alt-bg:#1b3346; --none-bg:#222d36; color-scheme:dark; }}
body {{ background:var(--paper); color:var(--ink); font-family:var(--f-body); font-size:15px; line-height:1.55; }}
.wrap {{ max-width:1180px; margin:0 auto; padding-inline:20px; padding-block:28px 64px; display:grid; gap:40px; }}
header {{ display:grid; gap:10px; border-bottom:2px solid var(--ink); padding-bottom:18px; }}
.eyebrow {{ font-family:var(--f-mono); font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-2); }}
h1 {{ font-family:var(--f-disp); font-weight:800; font-size:clamp(30px,5vw,46px); line-height:1.05; margin:0; letter-spacing:-.01em; text-wrap:balance; }}
h2 {{ font-family:var(--f-disp); font-weight:700; font-size:24px; margin:0; text-wrap:balance; }}
h3 {{ font-family:var(--f-disp); font-weight:700; font-size:17px; margin:0; }}
p {{ margin:0; max-width:72ch; }}
.lede {{ font-size:17px; color:var(--ink-2); max-width:70ch; }}
section {{ display:grid; gap:16px; min-width:0; }}
.verdict {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:14px; }}
.v {{ background:var(--panel); border:1px solid var(--rule); border-top:4px solid var(--c,var(--accent)); padding:14px 16px; display:grid; gap:6px; align-content:start; }}
.v.bad {{ --c:var(--loss); }} .v.ok {{ --c:var(--gain); }} .v.mid {{ --c:var(--warn); }}
.v .k {{ font-family:var(--f-mono); font-size:22px; font-weight:500; font-variant-numeric:tabular-nums; }}
.v p {{ font-size:14px; color:var(--ink-2); }}
.scroll {{ overflow-x:auto; border:1px solid var(--rule); background:var(--panel); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }}
th, td {{ padding:8px 10px; text-align:left; vertical-align:top; border-bottom:1px solid var(--rule); }}
thead th {{ font-family:var(--f-mono); font-weight:500; font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-2); background:var(--tint); white-space:nowrap; position:sticky; top:0; }}
tbody th {{ font-weight:600; }}
.opps td, .opps th {{ min-width:64px; }}
.opps .setup {{ min-width:170px; }} .opps .lv {{ min-width:160px; }}
.opps tr.main > * {{ border-bottom:none; }}
.opps tr.detail td {{ padding-top:0; font-size:12.5px; }}
.opps tr.detail .inner {{ position:sticky; left:10px; max-width:min(1020px, calc(100vw - 150px)); }} .opps tr.detail .sum {{ display:inline; margin-left:4px; }} .opps tr.detail .sub {{ display:inline; margin-left:10px; }}
.tk {{ font-family:var(--f-mono); font-size:14px; }}
.sc {{ font-family:var(--f-mono); font-size:14px; font-weight:500; }}
.sub {{ font-size:11.5px; color:var(--ink-2); font-weight:400; font-family:var(--f-body); }}
.sum {{ font-size:12.5px; margin-top:3px; }}
.pos {{ color:var(--gain); }} .neg {{ color:var(--loss); }} .muted {{ color:var(--ink-2); }}
.pill {{ display:inline-block; font-family:var(--f-mono); font-size:11.5px; font-weight:500; padding:2px 8px; border-radius:3px; white-space:nowrap; }}
.p-long {{ background:var(--long-bg); color:var(--gain); }} .p-short {{ background:var(--short-bg); color:var(--loss); }}
.p-alt {{ background:var(--alt-bg); color:var(--accent); }} .p-none {{ background:var(--none-bg); color:var(--ink-2); }}
.tag {{ font-family:var(--f-mono); font-size:10.5px; text-transform:uppercase; letter-spacing:.05em; padding:1px 5px; border:1px solid currentColor; border-radius:2px; }}
.t-fundamental {{ color:var(--gain); }} .t-speculative {{ color:var(--loss); }} .t-mixed {{ color:var(--warn); }} .t-unclear {{ color:var(--ink-2); }}
.bench th, .bench td {{ background:var(--tint); }}
.chart {{ background:var(--panel); border:1px solid var(--rule); padding:12px 12px 6px; display:grid; gap:8px; }}
.chart svg {{ width:100%; height:auto; display:block; }}
.grid {{ stroke:var(--rule); stroke-width:1; }} .axl {{ stroke:var(--ink-2); }}
.ax {{ fill:var(--ink-2); font:11px var(--f-mono); }} .lbl {{ fill:var(--accent); font:600 11px var(--f-mono); }}
.ho {{ fill:var(--ho); }}
.s-final {{ fill:none; stroke:var(--accent); stroke-width:2; }} .s-final-dot {{ fill:var(--accent); }}
.s-rules {{ fill:none; stroke:var(--warn); stroke-width:1.6; }} .s-rules-dot {{ fill:var(--warn); }}
.s-spy {{ fill:none; stroke:var(--ink-2); stroke-width:1.4; stroke-dasharray:4 3; }} .s-spy-dot {{ fill:var(--ink-2); }}
.legend {{ display:flex; flex-wrap:wrap; gap:16px; font-size:13px; }}
.lg {{ display:inline-flex; align-items:center; gap:6px; }} .lg i {{ width:18px; height:3px; display:inline-block; }}
.s-final-k {{ background:var(--accent); }} .s-rules-k {{ background:var(--warn); }} .s-spy-k {{ background:var(--ink-2); }}
.steps {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; }}
.steps > div {{ background:var(--panel); border:1px solid var(--rule); padding:12px 14px; display:grid; gap:4px; align-content:start; }}
.steps p {{ font-size:14px; }}
.two {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:24px; }}
.two > div {{ min-width:0; display:grid; gap:10px; align-content:start; }}
ul {{ margin:0; padding-left:20px; display:grid; gap:6px; max-width:76ch; }}
code {{ font-family:var(--f-mono); font-size:12.5px; background:var(--tint); padding:0 4px; border-radius:2px; }}
.note {{ font-size:13px; color:var(--ink-2); }}
dl {{ display:grid; grid-template-columns:max-content 1fr; gap:6px 16px; margin:0; font-size:14px; }}
dt {{ font-family:var(--f-mono); font-size:12.5px; color:var(--accent); }} dd {{ margin:0; }}
@media (max-width:560px) {{ dl {{ grid-template-columns:1fr; }} dd {{ margin-bottom:6px; }} }}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">Signals from Fri Oct 2, 2026 close · buy at Mon Oct 5 close · 471 liquid U.S. stocks scanned</div>
  <h1>Extreme Mover Screen</h1>
  <p class="lede">A daily scan for stocks with abnormal acceleration. The top section is the buy-at-close plan: buy three extreme movers at the closing price, sell into the first small pop. It held up on data it was never tuned on. Everything below it is the research behind the plan, including the 10-day swing models that did not hold up.</p>
</header>

<section>
  <h2>Monday's buy-at-close plan</h2>
  <div class="steps">
    <div><span class="eyebrow">Buy</span><p>Place a market-on-close buy for each stock any time Monday. It fills at Monday's closing price. Split your capital three ways.</p></div>
    <div><span class="eyebrow">Sell</span><p>From Tuesday on, sell as soon as the stock trades at the target: your fill price plus the target %. A limit sell at the target does this automatically. If it opens above the target, sell at the open.</p></div>
    <div><span class="eyebrow">Time limit</span><p>If the target never trades, sell at the close on Monday Oct 12, the fifth session. There is no stop. In testing, every stop lowered the average result.</p></div>
  </div>
  <div class="scroll"><table><thead><tr><th>Ticker</th><th>Fri close</th><th>Target</th><th>Sell price</th><th>20D</th><th>ATR</th><th>Why it's moving</th></tr></thead><tbody>{plan_rows}</tbody></table></div>
  <p class="note">The target is 0.25 × the stock's average daily range (ATR), measured from your actual fill. Recompute it from Monday's closing price. All three are chip stocks, so they will tend to win or lose together. If one of these is already in your account from an earlier signal, skip the duplicate.</p>
  <h3>How this rule did in the backtest</h3>
  <div class="scroll"><table><thead><tr><th>Period</th><th>Trades</th><th>Sold at a profit</th><th>Target hit day 1</th><th>Green at day-1 close</th><th>Average trade</th><th>Worst 5%</th><th>Profit factor</th><th>Return</th><th>Sharpe</th><th>Max drawdown</th></tr></thead><tbody>{plan_stat_rows()}</tbody></table></div>
  <p class="note">Rule fixed before the holdout was run. Costs: no spread on the closing-auction buy, 5–15 bps on the market sell. Return assumes all capital in up to three positions at a time. The high win rate comes from selling into small pops; a stock being green at the next close is still close to a coin flip. The cost is the rare large loss (worst cases −25% to −47% in five days) and drawdowns far deeper than SPY's. Daily bars cannot show whether a stock hit the target before or after other moves, so fills assume the target trades at exactly the target price.</p>
</section>

<section>
  <h2>Research: the 10-day swing models</h2>
  <div class="verdict">
    <div class="v bad"><div class="eyebrow">Final model, holdout Jul 2025 – Oct 2026</div><div class="k">{pct(hold['final']['equity']['total_return'],d=0)} vs SPY {pct(hold['SPY']['total_return'],d=0)}</div>
      <p>The design that looked strong in development (Sharpe {num(dev['final']['equity']['sharpe'])}) lost money once frozen and run on data it had not seen. Its momentum picks earned {pct(hold['final']['signals']['MOMENTUM']['avg'])} per trade, the same as buying every mover.</p></div>
    <div class="v bad"><div class="eyebrow">Reversal shorts</div><div class="k">{pct(full['baselines']['short_all_candidates']['avg'])} per trade</div>
      <p>Shorting extreme movers lost money in every sub-period, including after RSI ≥ 80 ({pct(full['baselines']['short_RSI80']['avg'])}) and after a confirmed breakdown ({pct(full['baselines']['short_all_confirmed_breaks']['avg'])}). Squeezes and borrow costs outweigh the pullbacks.</p></div>
    <div class="v ok"><div class="eyebrow">What held up</div><div class="k">Reversal Score AUC {auc['development']['rev_prob']:.2f} → {auc['holdout']['rev_prob']:.2f}</div>
      <p>The Reversal Score correctly ranks which movers suffer a large 5-day drop, with similar calibration in both periods. Use it to avoid chasing and to tighten stops, not as a short trigger.</p></div>
    <div class="v mid"><div class="eyebrow">Untuned rules variant, full OOS</div><div class="k">{pct(full['rules']['equity']['total_return'],d=0)} vs SPY {pct(full['SPY']['total_return'],d=0)}</div>
      <p>Positive in both periods, with more return than SPY but a lower Sharpe ({num(full['rules']['equity']['sharpe'])} vs {num(full['SPY']['sharpe'])}) and a deeper drawdown ({pct(full['rules']['equity']['max_drawdown'],False,0)}). Choosing it now, after seeing the holdout, would be hindsight.</p></div>
  </div>
</section>

<section>
  <h2>Extreme movers today, scored by the swing models</h2>
  <p class="note">LONG and SHORT come from the pre-registered final model. LONG* is a signal from the untuned rules variant only. Scores are 0–100: Extreme Move measures how abnormal the move is; Momentum and Reversal are walk-forward probabilities (in %) that a 10-day long or 5-day short nets more than 3% after costs. Their base rates are about 40% and 30%. Stops are checked on closing prices, to match close-only fills.</p>
  <div class="scroll"><table class="opps"><thead>{thead}</thead><tbody>{act_rows}</tbody></table></div>
  <h3>Watchlist: extreme movers with no trade today</h3>
  <p class="note">These moved enough to qualify but failed a gate. The setup column says which. A high Reversal Score here argues against chasing.</p>
  <div class="scroll"><table class="opps"><thead>{thead}</thead><tbody>{watch_rows}</tbody></table></div>
  <p class="note">Events inside the holding window: MRVL holds an investor day on Oct 6, AEHR reports around Oct 9, and FORM reports Oct 28 (just outside its 10-session hold). Catalyst notes come from news and SEC-filing research on Oct 3; sources are in the project files.</p>
</section>

<section>
  <h2>Walk-forward results</h2>
  <div class="chart">
    <div class="legend">{legend}</div>
    {''.join(svg)}
    <p class="note">Portfolio equity, out of sample, starting at 1.0 in July 2023. Up to 10 positions, each sized to risk 1.25% of equity at its stop, entries and exits at the close, 10–30 bps slippage per side plus borrow fees on shorts.</p>
  </div>
  <div class="scroll"><table><thead><tr><th>Strategy</th><th>Return</th><th>Sharpe</th><th>Max drawdown</th><th>3-month windows positive</th><th>Median 3-month</th><th>Worst 3-month</th></tr></thead>
    <tbody><tr><td colspan="7" class="eyebrow">Development · Jul 2023 – Jun 2025 (design choices made here)</td></tr>{strat_rows('development')}
    <tr><td colspan="7" class="eyebrow">Holdout · Jul 2025 – Oct 2026 (run once, untouched)</td></tr>{strat_rows('holdout')}</tbody></table></div>

  <div class="two">
    <div><h3>Momentum vs reversal trades, final model</h3>
    <div class="scroll"><table><thead><tr><th>Period</th><th>Side</th><th>Trades</th><th>Win rate</th><th>Average</th><th>Median</th><th>Profit factor</th><th>Sharpe*</th><th>Max DD*</th></tr></thead><tbody>{side_rows('final')}</tbody></table></div></div>
    <div><h3>Momentum vs reversal trades, rules variant</h3>
    <div class="scroll"><table><thead><tr><th>Period</th><th>Side</th><th>Trades</th><th>Win rate</th><th>Average</th><th>Median</th><th>Profit factor</th><th>Sharpe*</th><th>Max DD*</th></tr></thead><tbody>{side_rows('rules')}</tbody></table></div></div>
  </div>
  <p class="note">Trade figures are net per-trade returns across every signal. *Sharpe and max drawdown come from a portfolio holding only that side. Momentum trades are 10-session longs with a close below entry − 2 ATR as the stop. Reversal trades are 5-session shorts with a close above the spike high as the stop.</p>

  <div class="two">
    <div><h3>Simple baselines, net per trade</h3>
    <div class="scroll"><table><thead><tr><th>Rule</th><th>Dev n</th><th>Dev avg</th><th>Dev win</th><th>Holdout n</th><th>Holdout avg</th><th>Holdout win</th></tr></thead><tbody>{base_rows()}</tbody></table></div>
    <p class="note">Within extreme movers, buying fresh breakouts beat buying extended runs in development but not in the holdout. No short rule was reliably profitable.</p></div>
    <div><h3>Score calibration by quintile, predicted → actual %</h3>
    <div class="scroll"><table><thead><tr><th>Score</th><th colspan="2">Q1 dev · hold</th><th colspan="2">Q2</th><th colspan="2">Q3</th><th colspan="2">Q4</th><th colspan="2">Q5</th></tr></thead><tbody>{''.join(cal_rows)}</tbody></table></div>
    <p class="note">AUC, development → holdout: Reversal Score {auc['development']['rev_prob']:.2f} → {auc['holdout']['rev_prob']:.2f}; Momentum Score {auc['development']['mom_prob']:.2f} → {auc['holdout']['mom_prob']:.2f}; RSI alone as a reversal signal {auc['development']['rev_rsi']:.2f} → {auc['holdout']['rev_rsi']:.2f}. The Momentum Score inverted in the holdout, so treat it as uninformative.</p></div>
  </div>
</section>

<section>
  <h2>How the system works</h2>
  <div class="two">
    <div><h3>Data and features</h3>
    <ul>
      <li>{R['n_candidates_total']:,} extreme-move events from 488 stocks (the S&amp;P 500's more volatile sectors plus about 215 high-beta names: AI infrastructure, quantum, crypto miners and treasuries, space, nuclear, biotech, EVs, fintech, miners). Daily bars from Jan 2022 to Oct 2, 2026, via Robinhood.</li>
      <li>A stock is a candidate when it has a $3+ price, $15M+ average daily dollar volume, and any of +12% in 1 day, +20% in 3 or 5 days, +30% in 10 or 20 days, +50% in 60 days, or an Extreme Move Score of 40 or more.</li>
      <li>Features: 1/3/5/10/20/60-day returns and volatility-scaled z-scores; acceleration (slope differences); distance from the 20/50/200-day averages, 20-day VWAP and 52-week high; RSI, MACD and histogram fade; ATR and ATR expansion; realized volatility; Bollinger position; gap and intraday range; relative volume, volume acceleration and abnormal-volume z-score; share turnover; close location, wicks, failed highs, intraday reversals and breakdowns.</li>
      <li>Industry adjustment: returns versus the sector ETF and versus the average of same-sub-industry peers, a beta-adjusted residual (beta fit before the move), the share of the 20-day move explained by peers, and peer breadth. A move counts as company-specific when peers explain under 30% of it.</li>
    </ul></div>
    <div><h3>Scores and regimes</h3>
    <dl>
      <dt>Extreme Move</dt><dd>Weighted blend of the largest volatility-scaled move, the largest move against fixed thresholds (+20% in 1 day up to +150% in 60 days), climax volume, ATR expansion and the idiosyncratic z-score. Volume and volatility count only when the stock is actually up.</dd>
      <dt>Momentum</dt><dd>Regularized logistic model, refit monthly on the prior 18 months. Largest current weights: {coef_list(coefs_m)}.</dd>
      <dt>Reversal</dt><dd>Same method on the short outcome. Largest weights: {coef_list(coefs_r)}.</dd>
      <dt>MOMENTUM</dt><dd>Fresh breakout: 60-day gain ≤ 50%, accelerating, average close in the upper half of the range, Momentum Score above its base rate, not a blow-off.</dd>
      <dt>REVERSAL</dt><dd>Blow-off: 5-day gain ≥ 40%, a volume spike ≥ 5×, a failed high or long upper wicks, and a confirmed breakdown.</dd>
      <dt>NO TRADE</dt><dd>Everything else, with the failing gate stated.</dd>
    </dl></div>
  </div>
  <div class="two">
    <div><h3>Leakage controls</h3>
    <ul>
      <li>Features for day t use data through t's close; entry is the next session's close.</li>
      <li>Volatility, beta and volume baselines are measured before the move (lagged 20 sessions or one day) so a spike cannot normalize itself.</li>
      <li>Models are refit monthly on events whose full outcome window closed at least 16 sessions before the test month (purge plus embargo). Thresholds use training base rates only.</li>
      <li>Design choices were made on Jul 2023 – Jun 2025. The Jul 2025 – Oct 2026 holdout was run once and is reported as is.</li>
    </ul></div>
    <div><h3>Known limits</h3>
    <ul>
      <li>Survivorship bias: the universe is today's index members and today's listed names. That flatters longs (winners were added) and penalizes shorts (collapsed names are missing).</li>
      <li>Share turnover uses today's share count for every past date.</li>
      <li>Historical catalysts are proxied from gaps and volume. Real news was researched only for today's names.</li>
      <li>Borrow costs are tiered estimates (5–60% a year), and hard-to-borrow names may be unavailable to short.</li>
      <li>Ten tickers had no data (delisted or not on Robinhood). HUT and CORZ history was cut at corporate-action breaks.</li>
    </ul></div>
  </div>
</section>
</div>
"""
open(f"{ROOT}/report.html", "w").write(html_out)
print("written", len(html_out))
