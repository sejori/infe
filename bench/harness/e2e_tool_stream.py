#!/usr/bin/env python3
"""End-to-end streaming tool-call load client (stdlib only).

Opens N concurrent streaming chat-completion requests that carry a `tools`
schema and a prompt that induces a tool call, and records per-request TTFT,
inter-token latency (ITL) between SSE chunks, end-to-end latency, chunk
count and whether tool_call deltas were seen. Writes raw JSON.

Usage:
  python3 e2e_tool_stream.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-1.5B-Instruct \
      --arm stock --engine vllm --concurrency 8 64 256 --requests 3 --output out.json
`--requests` = rounds per concurrency level (each round opens `concurrency` streams).
"""
import argparse, json, statistics, sys, threading, time, urllib.request

TOOLS = [
  {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
  {"type": "function", "function": {"name": "get_time", "description": "Get the local time in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]
CITIES = ["London", "Paris", "Tokyo", "Berlin", "Madrid", "Rome", "Oslo", "Lima", "Cairo", "Delhi"]

def one_request(base_url, model, idx, max_tokens, timeout):
    city = CITIES[idx % len(CITIES)]
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
            "tools": TOOLS, "tool_choice": "auto",
            "messages": [{"role": "system", "content": "You are a helpful assistant. Always use the provided tools to answer."},
                         {"role": "user", "content": f"What is the weather in {city} right now, in celsius? Then tell me the local time there."}]}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; last = None; itls = []; chunks = 0; tool_deltas = 0; content_chars = 0; err = None
    acc = {}  # parity: accumulate tool calls by index -> {id, name, arguments}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"): continue
                payload = line[5:].strip()
                if payload == "[DONE]": break
                now = time.perf_counter()
                try: obj = json.loads(payload)
                except json.JSONDecodeError: continue
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                if not (delta.get("content") or delta.get("tool_calls") or delta.get("reasoning_content")): continue
                chunks += 1
                if delta.get("tool_calls"):
                    tool_deltas += len(delta["tool_calls"])
                    for tc in delta["tool_calls"]:
                        a = acc.setdefault(tc.get("index", 0), {"id": None, "name": None, "arguments": ""})
                        if tc.get("id"): a["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"): a["name"] = fn["name"]
                        if fn.get("arguments"): a["arguments"] += fn["arguments"]
                if delta.get("content"): content_chars += len(delta["content"])
                if first is None: first = now
                elif last is not None: itls.append((now - last) * 1000)
                last = now
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:200]
    t1 = time.perf_counter()
    parity = []
    for i, a in sorted(acc.items()):
        try: args = json.loads(a["arguments"]); ok = isinstance(args, dict) and "name" not in args
        except Exception: args = None; ok = False
        parity.append({"index": i, "has_id": bool(a["id"]), "name": a["name"], "args_json_ok": ok, "args_keys": sorted(args) if isinstance(args, dict) else None, "args_raw_len": len(a["arguments"])})
    return {"tool_calls_parity": parity, "idx": idx, "ttft_ms": (first - t0) * 1000 if first else None, "e2e_ms": (t1 - t0) * 1000,
            "chunks": chunks, "tool_deltas": tool_deltas, "content_chars": content_chars,
            "itl_ms": itls, "error": err}

def run_level(base_url, model, concurrency, max_tokens, timeout, offset):
    out = [None] * concurrency
    def worker(i): out[i] = one_request(base_url, model, offset + i, max_tokens, timeout)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t0 = time.perf_counter(); [t.start() for t in ts]; [t.join() for t in ts]
    return out, (time.perf_counter() - t0) * 1000

def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs) - 1) * p; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True); ap.add_argument("--engine", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[8, 64, 256])
    ap.add_argument("--requests", type=int, default=3, help="rounds per concurrency level")
    ap.add_argument("--max-tokens", type=int, default=160); ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--warmup", type=int, default=4); ap.add_argument("--output", required=True)
    a = ap.parse_args()
    run_level(a.base_url, a.model, a.warmup, a.max_tokens, a.timeout, 0)  # warmup, discarded
    report = {"arm": a.arm, "engine": a.engine, "model": a.model, "max_tokens": a.max_tokens,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "levels": []}
    for conc in a.concurrency:
        for rnd in range(a.requests):
            rows, wall = run_level(a.base_url, a.model, conc, a.max_tokens, a.timeout, rnd * conc)
            ok = [r for r in rows if not r["error"]]
            itl = [x for r in ok for x in r["itl_ms"]]
            summ = {"concurrency": conc, "round": rnd, "wall_ms": wall, "ok": len(ok), "errors": len(rows) - len(ok),
                    "ttft_p50": pct([r["ttft_ms"] for r in ok if r["ttft_ms"]], .5), "ttft_p99": pct([r["ttft_ms"] for r in ok if r["ttft_ms"]], .99),
                    "itl_p50": pct(itl, .5), "itl_p99": pct(itl, .99), "itl_mean": statistics.fmean(itl) if itl else None,
                    "e2e_p50": pct([r["e2e_ms"] for r in ok], .5), "chunks_total": sum(r["chunks"] for r in ok),
                    "tool_deltas_total": sum(r["tool_deltas"] for r in ok), "chunks_per_s": sum(r["chunks"] for r in ok) / (wall / 1000),
                    "parity_calls": sum(len(r["tool_calls_parity"]) for r in ok),
                    "parity_args_ok": sum(1 for r in ok for t in r["tool_calls_parity"] if t["args_json_ok"]),
                    "parity_has_id": sum(1 for r in ok for t in r["tool_calls_parity"] if t["has_id"]),
                    "parity_has_name": sum(1 for r in ok for t in r["tool_calls_parity"] if t["name"])}
            report["levels"].append({"summary": summ, "requests": rows})
            print(f"[{a.engine}/{a.arm}] conc={conc:4d} rnd={rnd} ok={len(ok)}/{len(rows)} ttft_p50={summ['ttft_p50'] and round(summ['ttft_p50'],1)}ms "
                  f"itl_p50={summ['itl_p50'] and round(summ['itl_p50'],2)}ms itl_p99={summ['itl_p99'] and round(summ['itl_p99'],2)}ms "
                  f"tool_deltas={summ['tool_deltas_total']} calls={summ['parity_calls']} args_ok={summ['parity_args_ok']} id={summ['parity_has_id']} name={summ['parity_has_name']} chunks/s={summ['chunks_per_s']:.0f}", flush=True)
            if len(ok) == 0: print("   first error:", rows[0]["error"], file=sys.stderr)
    with open(a.output, "w") as f: json.dump(report, f)
    print("wrote", a.output)

if __name__ == "__main__": main()
