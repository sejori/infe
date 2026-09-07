#!/usr/bin/env python3
"""Aggregate e2e_tool_stream.py reports into a per-engine stock-vs-infe table (median over rounds).

D6: now reports deltas-per-request per arm so ITL can be normalized by delta count.
"""
import glob, json, statistics, sys
rows = {}
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "*.json")):
    if "_ab.json" in f: continue
    r = json.load(open(f))
    try: cpu = [float(x.strip().rstrip('%')) for x in open(f[:-5] + ".cpu.txt") if x.strip()]
    except FileNotFoundError: cpu = []
    for lv in r["levels"]:
        s = lv["summary"]
        k = (r["engine"], r["arm"], s["concurrency"])
        d = rows.setdefault(k, {"ttft_p50": [], "itl_p50": [], "itl_p99": [], "e2e_p50": [], "chunks_per_s": [], "errors": 0, "calls": 0, "args_ok": 0, "has_id": 0, "cpu": cpu,
                                "tool_deltas_total": 0, "ok_requests": 0, "itl_per_delta": []})
        for m in ("ttft_p50", "itl_p50", "itl_p99", "e2e_p50", "chunks_per_s"):
            if s.get(m) is not None: d[m].append(s[m])
        d["errors"] += s["errors"]; d["calls"] += s["parity_calls"]; d["args_ok"] += s["parity_args_ok"]; d["has_id"] += s["parity_has_id"]
        d["tool_deltas_total"] += s.get("tool_deltas_total", 0)
        d["ok_requests"] += s.get("ok", 0)
        # D6: collect ITL values normalized by delta count per request
        for req in lv.get("requests", []):
            req_deltas = req.get("tool_deltas", 0)
            req_itls = req.get("itl_ms", [])
            if req_deltas > 0 and req_itls:
                d["itl_per_delta"].extend([x / req_deltas for x in req_itls])

def med(xs): return statistics.median(xs) if xs else float("nan")
def iqr(xs):
    if len(xs) < 2: return 0.0
    q = statistics.quantiles(xs, n=4); return q[2] - q[0]

print(f"{'engine':7} {'arm':6} {'conc':>4} {'ttft_p50':>9} {'itl_p50':>8} {'itl_p99':>8} {'e2e_p50':>8} {'chunks/s':>8} {'cpu%':>5} {'err':>4} {'calls':>5} {'args_ok':>7} {'has_id':>6} {'deltas/req':>10}")
for (e, a, c), d in sorted(rows.items()):
    dpr = d["tool_deltas_total"] / d["ok_requests"] if d["ok_requests"] > 0 else 0
    print(f"{e:7} {a:6} {c:4d} {med(d['ttft_p50']):9.1f} {med(d['itl_p50']):8.2f} {med(d['itl_p99']):8.1f} {med(d['e2e_p50']):8.0f} {med(d['chunks_per_s']):8.0f} {med(d['cpu']):5.0f} {d['errors']:4d} {d['calls']:5d} {d['args_ok']:7d} {d['has_id']:6d} {dpr:10.1f}")

print("\nstock→infe deltas (median of rounds; negative = infe faster):")
for (e, a, c), d in sorted(rows.items()):
    if a != "infe": continue
    s = rows.get((e, "stock", c))
    if not s: continue
    stock_dpr = s["tool_deltas_total"] / s["ok_requests"] if s["ok_requests"] > 0 else 0
    infe_dpr = d["tool_deltas_total"] / d["ok_requests"] if d["ok_requests"] > 0 else 0
    print(f"  {e:7} conc={c:3d} deltas/req: stock={stock_dpr:.1f} infe={infe_dpr:.1f} ratio={infe_dpr/stock_dpr:.2f}x" if stock_dpr > 0 else f"  {e:7} conc={c:3d} deltas/req: stock=0 infe={infe_dpr:.1f}")
    for m in ("ttft_p50", "itl_p50", "itl_p99", "e2e_p50"):
        sm, im = med(s[m]), med(d[m])
        print(f"  {e:7} conc={c:3d} {m:8} stock={sm:8.2f} (IQR {iqr(s[m]):.2f})  infe={im:8.2f} (IQR {iqr(d[m]):.2f})  Δ={100*(im-sm)/sm:+6.1f}%")

    # D6: normalized ITL (per-delta) comparison
    s_nitl = med(s["itl_per_delta"]) if s["itl_per_delta"] else None
    d_nitl = med(d["itl_per_delta"]) if d["itl_per_delta"] else None
    if s_nitl is not None and d_nitl is not None and s_nitl > 0:
        print(f"  {e:7} conc={c:3d} itl/delta stock={s_nitl:8.4f}  infe={d_nitl:8.4f}  Δ={100*(d_nitl-s_nitl)/s_nitl:+6.1f}%  (normalized)")
