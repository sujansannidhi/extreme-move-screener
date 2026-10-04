"""Convert a saved get_equity_fundamentals result to CSV rows (append mode).
Usage: python3 rh_fund_to_csv.py <saved_result_file> <out_csv>"""
import json, sys, csv, os
src, out = sys.argv[1], sys.argv[2]
raw = open(src, encoding="utf-8").read()
obj = json.loads(raw[raw.index("[") if raw.lstrip().startswith("[") else raw.index("{"):])
if isinstance(obj, list):
    obj = next(json.loads(b["text"]) for b in obj if isinstance(b, dict) and b.get("type") == "text")
cols = ["symbol", "shares_outstanding", "float", "market_cap", "pe_ratio", "pb_ratio", "industry", "sector", "average_volume_30_days"]
new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(cols)
    for r in obj["data"]["results"]:
        w.writerow([r.get(c) for c in cols])
print(len(obj["data"]["results"]), "rows appended to", out, "not_found:", obj["data"].get("not_found"))
