#!/usr/bin/env python3
"""High-admission workload for the infe-sched M0 probe.

Three workloads that stress the scheduler in ways e2e_tool_stream.py does not:

  --mode mixed_lengths    Varying prompt + output lengths to keep the prefill
                          admission path active every step (chunked prefill
                          budgets, length-based sorting, mixed prefill+decode)

  --mode short_burst      Many short requests arriving in a tight window
                          (admission + batching under queue pressure)

  --mode tool_call        The standard tool-call workload from
                          e2e_tool_stream.py (baseline / control)

Design notes:
  - Uses the same streaming SSE protocol and parity tracking as
    e2e_tool_stream.py so summarize_ab.py can ingest both.
  - Adds prefix_hit estimation via a shared_prefix config so we can
    correlate scheduling overhead with cache pressure.
  - max_tokens is varied per request in mixed_lengths mode to force the
    scheduler to batch decode steps with different completion times.
  - The prompt sizes vary from short (32 tokens) to long (2000+ tokens)
    to stress chunked prefill admission.
"""
import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.request

# ---------------------------------------------------------------------------
# Shared system prompt for prefix-cache pressure (varied length)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_SHORT = "You are a helpful assistant. Answer concisely."

_SYSTEM_PROMPT_LONG = (
    "You are a helpful assistant. " * 100  # ~500 tokens of shared prefix
)

_TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"},
                                      "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_time", "description": "Get the local time in a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
]

_CITIES = ["London", "Paris", "Tokyo", "Berlin", "Madrid", "Rome", "Oslo",
           "Lima", "Cairo", "Delhi", "Seoul", "Sydney", "Toronto", "Vienna",
           "Athens", "Dublin", "Lisbon", "Warsaw", "Prague", "Budapest"]

_QUESTIONS = [
    "What is the weather in {city} right now, in celsius?",
    "Tell me the local time in {city}.",
    "What's the weather in {city} and what time is it there?",
    "I need the current temperature in {city} in fahrenheit.",
    "Is it raining in {city} right now? Also what time is it there?",
]


def _make_request(base_url, model, idx, mode, max_tokens, shared_prefix, timeout):
    """Build and send one streaming request, return result dict."""
    city = _CITIES[idx % len(_CITIES)]
    question = _QUESTIONS[idx % len(_QUESTIONS)].format(city=city)

    if shared_prefix == "long":
        sys_prompt = _SYSTEM_PROMPT_LONG
    else:
        sys_prompt = _SYSTEM_PROMPT_SHORT

    # In mixed_lengths mode, vary max_tokens per request
    if mode == "mixed_lengths":
        mtok = random.choice([32, 64, 96, 128, 160, 200])
    else:
        mtok = max_tokens

    body = {
        "model": model, "stream": True, "max_tokens": mtok, "temperature": 0,
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": question}],
    }
    # Add tools only for tool_call mode
    if mode == "tool_call":
        body["tools"] = _TOOLS
        body["tool_choice"] = "auto"

    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    last = None
    itls = []
    chunks = 0
    tool_deltas = 0
    content_chars = 0
    err = None
    acc = {}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                now = time.perf_counter()
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                if not (delta.get("content") or delta.get("tool_calls")
                        or delta.get("reasoning_content")):
                    continue
                chunks += 1
                if delta.get("tool_calls"):
                    tool_deltas += len(delta["tool_calls"])
                    for tc in delta["tool_calls"]:
                        a = acc.setdefault(tc.get("index", 0),
                                          {"id": None, "name": None, "arguments": ""})
                        if tc.get("id"):
                            a["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            a["name"] = fn["name"]
                        if fn.get("arguments"):
                            a["arguments"] += fn["arguments"]
                if delta.get("content"):
                    content_chars += len(delta["content"])
                if first is None:
                    first = now
                elif last is not None:
                    itls.append((now - last) * 1000)
                last = now
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:200]
    t1 = time.perf_counter()

    parity = []
    for i, a in sorted(acc.items()):
        try:
            args = json.loads(a["arguments"])
            ok = isinstance(args, dict) and "name" not in args
        except Exception:
            args = None
            ok = False
        parity.append({"index": i, "has_id": bool(a["id"]), "name": a["name"],
                       "args_json_ok": ok, "args_raw_len": len(a["arguments"])})
    return {
        "tool_calls_parity": parity, "idx": idx,
        "ttft_ms": (first - t0) * 1000 if first else None,
        "e2e_ms": (t1 - t0) * 1000, "chunks": chunks,
        "tool_deltas": tool_deltas, "content_chars": content_chars,
        "itl_ms": itls, "error": err, "max_tokens_requested": mtok,
    }


def _run_level(base_url, model, concurrency, mode, max_tokens, shared_prefix,
               timeout, offset):
    out = [None] * concurrency

    def worker(i):
        out[i] = _make_request(base_url, model, offset + i, mode, max_tokens,
                               shared_prefix, timeout)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out, (time.perf_counter() - t0) * 1000


def _pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser(
        description="High-admission workload for infe-sched M0 probe"
    )
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, help="Arm label (stock, probe, etc.)")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--mode", choices=["mixed_lengths", "short_burst", "tool_call"],
                    default="mixed_lengths")
    ap.add_argument("--shared-prefix", choices=["short", "long"], default="short",
                    help="Use a longer shared system prompt to increase prefix-cache "
                         "pressure (short ≈ 10 tokens, long ≈ 500 tokens)")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[8, 64, 256])
    ap.add_argument("--requests", type=int, default=3,
                    help="Rounds per concurrency level")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    random.seed(42)  # deterministic request lengths for repeated runs

    # Warmup (discarded)
    _run_level(a.base_url, a.model, a.warmup, a.mode, a.max_tokens,
               a.shared_prefix, a.timeout, 0)

    report = {
        "arm": a.arm, "engine": a.engine, "model": a.model,
        "mode": a.mode, "shared_prefix": a.shared_prefix,
        "max_tokens": a.max_tokens,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "levels": [],
    }

    for conc in a.concurrency:
        for rnd in range(a.requests):
            rows, wall = _run_level(
                a.base_url, a.model, conc, a.mode, a.max_tokens,
                a.shared_prefix, a.timeout, rnd * conc
            )
            ok = [r for r in rows if not r["error"]]
            itl = [x for r in ok for x in r["itl_ms"]]
            summ = {
                "concurrency": conc, "round": rnd, "wall_ms": wall,
                "ok": len(ok), "errors": len(rows) - len(ok),
                "ttft_p50": _pct([r["ttft_ms"] for r in ok if r["ttft_ms"]], .5),
                "ttft_p99": _pct([r["ttft_ms"] for r in ok if r["ttft_ms"]], .99),
                "itl_p50": _pct(itl, .5), "itl_p99": _pct(itl, .99),
                "itl_mean": statistics.fmean(itl) if itl else None,
                "e2e_p50": _pct([r["e2e_ms"] for r in ok], .5),
                "chunks_total": sum(r["chunks"] for r in ok),
                "tool_deltas_total": sum(r["tool_deltas"] for r in ok),
                "chunks_per_s": sum(r["chunks"] for r in ok) / (wall / 1000),
                "parity_calls": sum(len(r["tool_calls_parity"]) for r in ok),
                "parity_args_ok": sum(1 for r in ok for t in r["tool_calls_parity"]
                                       if t["args_json_ok"]),
                "parity_has_id": sum(1 for r in ok for t in r["tool_calls_parity"]
                                     if t["has_id"]),
                "parity_has_name": sum(1 for r in ok for t in r["tool_calls_parity"]
                                       if t["name"]),
            }
            report["levels"].append({"summary": summ, "requests": rows})
            mean_mtok = statistics.mean(
                r["max_tokens_requested"] for r in ok
            ) if ok else 0
            print(
                f"[{a.engine}/{a.arm}] mode={a.mode} conc={conc:4d} rnd={rnd} "
                f"ok={len(ok)}/{len(rows)} ttft_p50={summ['ttft_p50'] and round(summ['ttft_p50'], 1)}ms "
                f"itl_p50={summ['itl_p50'] and round(summ['itl_p50'], 2)}ms "
                f"e2e_p50={summ['e2e_p50'] and round(summ['e2e_p50'], 0)}ms "
                f"mean_maxtok={mean_mtok:.0f} "
                f"chunks/s={summ['chunks_per_s']:.0f}",
                flush=True,
            )
            if len(ok) == 0:
                print("   first error:", rows[0]["error"], file=sys.stderr)

    with open(a.output, "w") as f:
        json.dump(report, f)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
