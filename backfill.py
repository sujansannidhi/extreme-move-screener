"""Write out/backfill_plans.csv: what the live rule would have published on each of the last N sessions before
launch. Clearly SIMULATED (it uses today's rule and today's universe), so the trade log has history to show."""
import sys

import pandas as pd

from features import ROOT, load_meta
from strategy import load_params
from daily import build_plan

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
LAUNCH = pd.Timestamp("2026-10-02")   # first LIVE signal date

meta = load_meta()
f = pd.read_pickle(f"{ROOT}/data/features.pkl")
f = f[f.symbol.isin(meta.query("group != 'etf'").index)]
days = sorted(d for d in f.date.unique() if d < LAUNCH)[-N:]
params = load_params()
plans = [build_plan(f[f.date == d], pd.Timestamp(d), params) for d in days]
out = pd.concat(plans, ignore_index=True)
out.to_csv(f"{ROOT}/out/backfill_plans.csv", index=False)
print(len(out), "simulated orders from", days[0].date(), "to", days[-1].date())
