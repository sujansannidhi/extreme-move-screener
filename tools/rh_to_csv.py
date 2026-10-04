"""Convert a saved Robinhood get_equity_historicals tool result into the project CSV format.
Usage: python3 rh_to_csv.py <saved_result_file> <out_csv>
Handles raw JSON, or a JSON list of content blocks [{"type":"text","text":"<json>"}]."""
import json, sys, csv


def load(path):
    raw = open(path, encoding="utf-8").read()
    obj = json.loads(raw[raw.index("[") if raw.lstrip().startswith("[") else raw.index("{"):])
    if isinstance(obj, list):  # content blocks
        for blk in obj:
            if isinstance(blk, dict) and blk.get("type") == "text":
                try:
                    return json.loads(blk["text"])
                except Exception:
                    continue
        raise ValueError("no JSON text block found")
    if isinstance(obj, dict) and "data" not in obj and "text" in obj:
        return json.loads(obj["text"])
    return obj


def fmt(x):
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    return s


def main(src, out):
    obj = load(src)
    results = obj["data"]["results"]
    n, syms = 0, []
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "date", "open", "high", "low", "close", "volume"])
        for r in results:
            bars = [b for b in (r.get("bars") or []) if not b.get("interpolated")]
            if bars:
                syms.append(r["symbol"])
            for b in bars:
                w.writerow([r["symbol"], b["begins_at"][:10], fmt(b["open_price"]), fmt(b["high_price"]),
                            fmt(b["low_price"]), fmt(b["close_price"]), int(b["volume"])])
                n += 1
    print(f"{len(syms)} symbols, {n} rows -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
